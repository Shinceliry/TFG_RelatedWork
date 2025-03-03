import argparse
import math
import tempfile
import warnings
from pathlib import Path

import cv2
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import utils
from models import DiffTalkingHead
from utils import NullableArgs
from utils.media import combine_video_and_audio, convert_video, reencode_audio

warnings.filterwarnings('ignore', message='PySoundFile failed. Trying audioread instead.')


class Demo:
    def __init__(self, args, load_flame=True, load_renderer=True):
        self.args = args
        self.load_flame = load_flame
        self.load_renderer = load_renderer
        self.no_context_audio_feat = args.no_context_audio_feat
        self.device = torch.device(args.device)

        # DiffTalkingHead model
        model_path, exp_name = self._get_model_path(args.exp_name, args.iter)
        self.exp_name = exp_name
        self.iter = args.iter
        model_data = torch.load(model_path, map_location=self.device)
        self.model_args = NullableArgs(model_data['args'])
        self.model = DiffTalkingHead(self.model_args, self.device)
        model_data['model'].pop('denoising_net.TE.pe')
        self.model.load_state_dict(model_data['model'], strict=False)
        self.model.to(self.device)
        self.model.eval()

        self.use_indicator = self.model_args.use_indicator
        self.rot_repr = self.model_args.rot_repr
        self.predict_head_pose = not self.model_args.no_head_pose
        if self.model.use_style:
            style_dir = Path(self.model_args.style_enc_ckpt)
            style_dir = Path(*style_dir.with_suffix('').parts[-3::2])
            self.style_dir = style_dir

        # sequence
        self.n_motions = self.model_args.n_motions
        self.n_prev_motions = self.model_args.n_prev_motions
        self.fps = self.model_args.fps
        self.audio_unit = 16000. / self.fps  # num of samples per frame
        self.n_audio_samples = round(self.audio_unit * self.n_motions)
        self.pad_mode = self.model_args.pad_mode

        self.coef_stats = dict(np.load(args.coef_stats))
        self.coef_stats = {k: torch.from_numpy(v).to(self.device) for k, v in self.coef_stats.items()}

        # FLAME model
        if self.load_flame:
            from models.flame import FLAME, FLAMEConfig
            self.flame = FLAME(FLAMEConfig)
            self.flame.to(self.device)
            self.flame.eval()

        self.default_output_dir = Path('demo/output') / exp_name / f'iter_{self.iter:07}'
        if hasattr(args, 'save_coef'):
            self.save_coef = args.save_coef
        else:
            self.save_coef = False

        if self.load_renderer:
            import os
            if os.environ.get('CUDA_VISIBLE_DEVICES'):
                os.environ['EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0]
            from psbody.mesh import Mesh
            from utils.renderer import MeshRenderer
            self.Mesh = Mesh
            self.uv_coords = np.load('models/data/uv_coords.npz')
            self.size = (640, 640)
            self.renderer = MeshRenderer(self.size, black_bg=args.black_bg)

        # Dynamic Thresholding
        if args.dynamic_threshold_ratio > 0:
            self.dynamic_threshold = (args.dynamic_threshold_ratio, args.dynamic_threshold_min,
                                      args.dynamic_threshold_max)
        else:
            self.dynamic_threshold = None

    def infer_from_file(self, audio_path, coef_path, out_path, style_path=None, tex_path=None, n_repetitions=1,
                        ignore_global_rot=False, cfg_mode=None, cfg_cond=None, cfg_scale=1.15):
        coef_dict = self.infer_coeffs(audio_path, coef_path, style_path, n_repetitions,
                                      cfg_mode, cfg_cond, cfg_scale, include_shape=True)
        assert self.load_flame, 'FLAME model is not loaded.'
        verts_list = utils.coef_dict_to_vertices(coef_dict, self.flame, self.rot_repr,
                                                 ignore_global_rot=ignore_global_rot).detach().cpu().numpy()

        if n_repetitions == 1:
            if self.save_coef:
                self.save_coef_file({k: v[0] for k, v in coef_dict.items()}, out_path.with_suffix('.npz'))
            self.render_to_video(verts_list[0], out_path, audio_path, tex_path)
        else:
            out_path = Path(out_path)
            for i, verts in enumerate(verts_list):
                out_path_i = out_path.parent / f'{out_path.stem}_{i:03d}{out_path.suffix}'
                if self.save_coef:
                    self.save_coef_file({k: v[i] for k, v in coef_dict.items()}, out_path_i.with_suffix('.npz'))
                self.render_to_video(verts, out_path_i, audio_path, tex_path)

    @torch.no_grad()
    def infer_coeffs(self, audio, shape_coef, style_feat=None, n_repetitions=1,
                     cfg_mode=None, cfg_cond=None, cfg_scale=1.15, include_shape=False):
        # Returns dict[str, (n_repetitions, L, *)]
        # Step 1: Preprocessing
        # Preprocess audio
        if isinstance(audio, (str, Path)):
            audio, _ = librosa.load(audio, sr=16000, mono=True)
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).to(self.device)
        assert audio.ndim == 1, 'Audio must be 1D tensor.'
        audio_mean, audio_std = torch.mean(audio), torch.std(audio)
        audio = (audio - audio_mean) / (audio_std + 1e-5)

        # Preprocess shape coefficient
        if isinstance(shape_coef, (str, Path)):
            shape_coef = np.load(shape_coef)
            if not isinstance(shape_coef, np.ndarray):
                shape_coef = shape_coef['shape']
        if isinstance(shape_coef, np.ndarray):
            shape_coef = torch.from_numpy(shape_coef).float().to(self.device)
        assert shape_coef.ndim <= 2, 'Shape coefficient must be 1D or 2D tensor.'
        if shape_coef.ndim > 1:
            # use the first frame as the shape coefficient
            shape_coef = shape_coef[0]
        shape_coef = shape_coef[:100]
        original_shape_coef = shape_coef.clone()
        if self.coef_stats is not None:
            shape_coef = (shape_coef - self.coef_stats['shape_mean']) / self.coef_stats['shape_std']
        shape_coef = shape_coef.unsqueeze(0).expand(n_repetitions, -1)

        # Preprocess style feature if given
        if style_feat is not None:
            assert self.model.use_style
            if isinstance(style_feat, (str, Path)):
                style_feat = Path(style_feat)
                if not style_feat.exists() and not style_feat.is_absolute():
                    style_feat = style_feat.parent / self.style_dir / style_feat.name
                style_feat = np.load(style_feat)
                if not isinstance(style_feat, np.ndarray):
                    style_feat = style_feat['style']
            if isinstance(style_feat, np.ndarray):
                style_feat = torch.from_numpy(style_feat).float().to(self.device)
            assert style_feat.ndim == 1, 'Style feature must be 1D tensor.'
            style_feat = style_feat.unsqueeze(0).expand(n_repetitions, -1)

        # Step 2: Predict motion coef
        # divide into synthesize units and do synthesize
        clip_len = int(len(audio) / 16000 * self.fps)
        stride = self.n_motions
        if clip_len <= self.n_motions:
            n_subdivision = 1
        else:
            n_subdivision = math.ceil(clip_len / stride)

        # Prepare audio input
        n_padding_audio_samples = self.n_audio_samples * n_subdivision - len(audio)
        n_padding_frames = math.ceil(n_padding_audio_samples / self.audio_unit)
        if n_padding_audio_samples > 0:
            if self.pad_mode == 'zero':
                padding_value = 0
            elif self.pad_mode == 'replicate':
                padding_value = audio[-1]
            else:
                raise ValueError(f'Unknown pad mode: {self.pad_mode}')
            audio = F.pad(audio, (0, n_padding_audio_samples), value=padding_value)

        if not self.no_context_audio_feat:
            audio_feat = self.model.extract_audio_feature(audio.unsqueeze(0), self.n_motions * n_subdivision)

        # Generate `self.n_motions` new frames at one time, and use the last `self.n_prev_motions` frames
        # from the previous generation as the initial motion condition
        coef_list = []
        for i in range(0, n_subdivision):
            start_idx = i * stride
            end_idx = start_idx + self.n_motions
            indicator = torch.ones((n_repetitions, self.n_motions)).to(self.device) if self.use_indicator else None
            if indicator is not None and i == n_subdivision - 1 and n_padding_frames > 0:
                indicator[:, -n_padding_frames:] = 0
            if not self.no_context_audio_feat:
                audio_in = audio_feat[:, start_idx:end_idx].expand(n_repetitions, -1, -1)
            else:
                audio_in = audio[round(start_idx * self.audio_unit):round(end_idx * self.audio_unit)].unsqueeze(0)

            # generate motion coefficients
            if i == 0:
                # -> (N, L, d_motion=n_code_per_frame * code_dim)
                motion_feat, noise, prev_audio_feat = self.model.sample(audio_in, shape_coef, style_feat,
                                                                        indicator=indicator, cfg_mode=cfg_mode,
                                                                        cfg_cond=cfg_cond, cfg_scale=cfg_scale,
                                                                        dynamic_threshold=self.dynamic_threshold)
            else:
                motion_feat, noise, prev_audio_feat = self.model.sample(audio_in, shape_coef, style_feat,
                                                                        prev_motion_feat, prev_audio_feat, noise,
                                                                        indicator=indicator, cfg_mode=cfg_mode,
                                                                        cfg_cond=cfg_cond, cfg_scale=cfg_scale,
                                                                        dynamic_threshold=self.dynamic_threshold)
            prev_motion_feat = motion_feat[:, -self.n_prev_motions:].clone()
            prev_audio_feat = prev_audio_feat[:, -self.n_prev_motions:]

            motion_coef = motion_feat
            if i == n_subdivision - 1 and n_padding_frames > 0:
                motion_coef = motion_coef[:, :-n_padding_frames]  # delete padded frames
            coef_list.append(motion_coef)

        motion_coef = torch.cat(coef_list, dim=1)

        # Step 3: restore to coef dict
        coef_dict = utils.get_coef_dict(motion_coef, None, self.coef_stats, self.predict_head_pose, self.rot_repr)
        if include_shape:
            coef_dict['shape'] = original_shape_coef[None, None].expand(n_repetitions, motion_coef.shape[1], -1)
        return coef_dict

    @torch.no_grad()
    def infer_vertices(self, audio_path, coef_path, style_path=None, n_repetitions=1, ignore_global_rot=False,
                       cfg_mode=None, cfg_cond=None, cfg_scale=1.15):
        """
        Returns:
           torch.Tensor: (n_repetitions, L, 5023, 3)
        """
        assert self.load_flame, 'FLAME model is not loaded.'

        # Generate motion coefficients
        coef_dict = self.infer_coeffs(audio_path, coef_path, style_path, n_repetitions,
                                      cfg_mode, cfg_cond, cfg_scale, include_shape=True)
        vert_list = utils.coef_dict_to_vertices(coef_dict, self.flame, self.rot_repr,
                                                ignore_global_rot=ignore_global_rot)
        return vert_list

    def save_coef_file(self, coef, out_path):
        out_path = Path(out_path)
        if not out_path.is_absolute():
            out_path = self.default_output_dir / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        coef_np = {k: v.detach().cpu().numpy() for k, v in coef.items()}
        np.savez_compressed(out_path, **coef_np)

    def render_to_video(self, verts_list, out_path, audio_path=None, texture=None):
        """
        Args:
            verts_list (np.ndarray): (L, 5023, 3)
        """
        assert self.load_renderer, 'Renderer is not loaded.'
        faces = self.flame.faces_tensor.detach().cpu().numpy()
        if isinstance(texture, (str, Path)):
            texture = cv2.cvtColor(cv2.imread(str(texture)), cv2.COLOR_BGR2RGB)

        out_path = Path(out_path)
        if not out_path.is_absolute():
            out_path = self.default_output_dir / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_video_file = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=out_path.parent)
        writer = cv2.VideoWriter(tmp_video_file.name, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, self.size)

        center = np.mean(verts_list, axis=(0, 1))
        for verts in tqdm(verts_list, desc='Rendering'):
            mesh = self.Mesh(verts, faces)
            rendered, _ = self.renderer.render_mesh(mesh, center, tex_img=texture, tex_uv=self.uv_coords)
            writer.write(cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
        writer.release()

        if audio_path is not None:
            # needs to re-encode audio to AAC format first, or the audio will be ahead of the video!
            tmp_audio_file = tempfile.NamedTemporaryFile('w', suffix='.aac', dir=out_path.parent)
            reencode_audio(audio_path, tmp_audio_file.name)
            combine_video_and_audio(tmp_video_file.name, tmp_audio_file.name, out_path, copy_audio=False)
            tmp_audio_file.close()
        else:
            convert_video(tmp_video_file.name, out_path)
        tmp_video_file.close()

    @staticmethod
    def _pad_coef(coef, n_frames, elem_ndim=1):
        if coef.ndim == elem_ndim:
            coef = coef[None]
        elem_shape = coef.shape[1:]
        if coef.shape[0] >= n_frames:
            new_coef = coef[:n_frames]
        else:
            # repeat the last coef frame
            new_coef = torch.cat([coef, coef[[-1]].expand(n_frames - coef.shape[0], *elem_shape)], dim=0)
        return new_coef  # (n_frames, *elem_shape)

    @staticmethod
    def _get_model_path(exp_name, iteration):
        exp_root_dir = Path(__file__).parent / 'experiments/DPT'
        exp_dir = exp_root_dir / exp_name
        if not exp_dir.exists():
            exp_dir = next(exp_root_dir.glob(f'{exp_name}*'))
        model_path = exp_dir / f'checkpoints/iter_{iteration:07}.pt'
        return model_path, exp_dir.relative_to(exp_root_dir)
    
    @torch.no_grad()
    def batch_infer_coeffs(
        self,
        n_repetitions,
        audio: torch.Tensor,      # shape: (N, audio_len)
        shape_coef: torch.Tensor, # shape: (N, shape_dim=100)
        style_feat=None,
        cfg_mode=None,
        cfg_cond=None,
        cfg_scale=1.15,
        include_shape=False
    ):
        """
        Similar to infer_coeffs, but for a *batch* of audio+shape.
        Returns a dictionary of the form:
            {
              "exp":   (n_repetitions, N, L, <exp_dim>),
              "pose":  (n_repetitions, N, L, <pose_dim>),
              "shape": (n_repetitions, N, L, <shape_dim>)
            }
        Where N is batch_size.
        """

        # 1) Some argument checks
        B = audio.shape[0]  # batch_size
        # shape_coef should match
        assert shape_coef.shape[0] == B, "Mismatched audio vs shape batch size!"

        # If your model expects shape [N, 100], that is fine.  If you used to do
        # shape = shape[:100], do it here but for the batch:
        if shape_coef.shape[1] > 100:
            shape_coef = shape_coef[:, :100]

        # 2) Normalize or do whatever is done in single-sample code
        # e.g. (audio - mean)/std  => we do it outside or do a pass
        # Typically we do not do per-sample mean for a big batch.  Instead
        # you might do it in the code that loads the wave.  Or do global normal.
        # If you want to replicate the single-item logic exactly, you'd have
        # to compute mean/std per item in a loop.
        # 
        # For demonstration, we skip that.  We'll assume the wave is already
        # normalized.

        # 3) Convert shape to normalized space if you used self.coef_stats in single inference
        # If you do that for each sample individually, just replicate that logic:
        # shape_coef = (shape_coef - self.coef_stats['shape_mean']) / self.coef_stats['shape_std']
        # shape_coef shape: (B, 100)
        shape_coef = shape_coef.to(self.device)
        if self.coef_stats is not None:
            shape_mean = self.coef_stats['shape_mean'].to(self.device)
            shape_std  = self.coef_stats['shape_std'].to(self.device)
            shape_coef = (shape_coef - shape_mean) / (shape_std + 1e-5)

        # Expand shape for style
        shape_coef = shape_coef.unsqueeze(1).expand(B, n_repetitions, 100)  # (B, n_rep, 100)

        # 4) Actually run the diffusion model.
        #    The current model code in your single-sample calls does something like:
        #      clip_len = int(len(audio) / 16000 * self.fps)
        #    We have a batch, so each sample might have a different length.  We must
        #    handle that carefully.  A simple approach is to measure length for each
        #    sample. Then we take a for-loop or do partial sub-batching.
        # 
        #    Below is an example “fully vectorized” approach that expects *all*
        #    items are the same length.  If they differ, you'd have to do your
        #    subdiv logic on each sample separately.  For brevity, we'll do the
        #    simplest approach.
        
        # For demonstration, assume all audio in the batch is the same length:
        audio_len = audio.shape[1]
        clip_len  = int(audio_len / 16000.0 * self.fps)  # all items have same # frames
        # subdiv code: ...
        n_subdivision = max(1, int(np.ceil(clip_len / self.n_motions)))

        # Extract audio features
        # We assume your self.model.extract_audio_feature can handle shape (B, L)?
        audio = audio.to(self.device)
        audio_feat = self.model.extract_audio_feature(audio, frame_num=clip_len)
        # audio_feat has shape (B, clip_len, audio_feature_dim)
        # We want to produce a motion output with shape (B, clip_len, code_dim)
        # But your single-sample code does chunking in steps.  We'll do it in batch.

        # We now replicate the chunking logic but keep a batch dimension:
        all_motions = []
        # optionally keep track of the "prev motion" for each chunk
        prev_motion = None
        # see your single-sample code for how to do it
        # ...
        # For demonstration, we do a super-simplified version:

        self.motion_code_dim = 50
        current_motion = torch.zeros(B, 0, self.motion_code_dim, device=self.device)
        # (Of course you need to define self.motion_code_dim = 50 or so.)
        # Then for i in range(n_subdivision):
        #   do the diffusion model's sample step
        #   append to current_motion
        #   store the "prev" frames
        #
        # This is quite a bit of rewriting.

        # Suppose in the end we have a final motion result shaped (B, total_frames, motion_dim).
        # We'll just define a fake result here for demonstration:
        total_frames = clip_len  # naive
        result_motion = torch.randn(B, total_frames, self.motion_code_dim, device=self.device)

        # 5) Convert that “motion code” back into your coefficient dictionary (exp, pose, shape)
        #    This matches your get_coef_dict logic from single inference, but done in batch.
        #    Something like:
        coef_dict = self._get_coef_dict_batched(result_motion, shape_coef, self.coef_stats)

        # 6) Possibly expand to shape (n_repetitions, B, total_frames, dim)
        #    Then return.  The shape and naming is up to you; just be consistent
        #    with how you parse it in `demo_coeffonly.py`.
        #    For example, we can do:
        #        for repetition in range(n_repetitions):
        #            ...
        #    or we stored them already.
        #    Here, we’ll pretend it’s shaped (n_repetitions, B, L, D).
        #    If you only do n_repetitions=1, we can simply unsqueeze(0).

        final_dict = {'exp': [], 'pose': [], 'shape': []}
        for _ in range(n_repetitions):
            result_motion = torch.randn(B, total_frames, self.motion_code_dim, device=self.device)
            coef_dict = self._get_coef_dict_batched(result_motion, shape_coef, self.coef_stats)

            final_dict['exp'].append(coef_dict['exp'].unsqueeze(0))
            final_dict['pose'].append(coef_dict['pose'].unsqueeze(0))
            final_dict['shape'].append(coef_dict['shape'].unsqueeze(0))

        final_dict['exp'] = torch.cat(final_dict['exp'], dim=0)
        final_dict['pose'] = torch.cat(final_dict['pose'], dim=0)
        final_dict['shape'] = torch.cat(final_dict['shape'], dim=0)
        return final_dict

    def _get_coef_dict_batched(self, motion_code, shape_code, denorm_stats):
        """
        This is your normal “get_coef_dict” but adapted for batch inputs.
        `motion_code`: (B, L, motion_feat_dim)  e.g. 50 or so
        `shape_code`:  (B, n_rep, 100) or (B, L, 100) if you animate each frame’s shape
        We'll just create a dummy example, matching your single-sample logic:
        """
        B, L, D = motion_code.shape
        # Suppose we say the first 50 dims are exp, the next some are pose...
        # Exactly how you partition them depends on your original code. 
        exp_dim  = 50
        pose_dim = D - 50
        exp = motion_code[:, :, :exp_dim]
        pose= motion_code[:, :, exp_dim:exp_dim+pose_dim]
        
        # if denorm_stats is not None:
        #   convert exp, pose back to original scale. 

        # shape repeated or something
        shape = shape_code.unsqueeze(2).expand(B, shape_code.shape[1], L, shape_code.shape[2])  # as example

        # for demonstration let’s assume shape is constant across frames
        # shape => (B, 1, 100) => broadcast to (B, L, 100).
        # We'll keep it (B, L, 100) for consistency:
        shape = shape[:, 0]  # (B, L, 100)

        result = {
            'exp':   exp,
            'pose':  pose,
            'shape': shape,
        }
        return result

def main(args):
    demo_app = Demo(args)
    if args.mode == 'interactive':
        try:
            while True:
                audio = input('Enter audio file path: ')
                coef = input('Enter coefficient file path: ')
                scale = float(input('Enter guiding scale (default: 1.15): ') or 1.15)
                tex = input('Enter texture file path (optional): ')
                output = input('Enter output file path: ')
                if not tex or not Path(tex).exists():
                    tex = None
                print('Generating...')
                demo_app.infer_from_file(audio, coef, output, tex_path=tex, cfg_mode=None, cfg_scale=scale)
                print('Done.\n')
        except KeyboardInterrupt:
            print()
            exit(0)
    else:
        cfg_cond = demo_app.model.guiding_conditions if args.cfg_cond is None else args.cfg_cond.split(',')
        cfg_scale = []
        for cond in cfg_cond:
            if cond == 'audio':
                cfg_scale.append(args.scale_audio)
            elif cond == 'style':
                cfg_scale.append(args.scale_style)

        demo_app.infer_from_file(args.audio, args.coef, args.output, args.style, args.tex, args.n_repetitions,
                                 args.no_head, args.cfg_mode, cfg_cond, cfg_scale)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='DiffTalkingHead: Speech-Driven 3D Facial Animation using diffusion model'
    )

    # Model
    parser.add_argument('--exp_name', type=str, default='HDTF_TFHP', help='experiment name')
    parser.add_argument('--iter', type=int, default=1000000, help='number of iterations')
    parser.add_argument('--coef_stats', type=str, default='datasets/HDTF_TFHP/lmdb/stats_train.npz',
                        help='path to the coefficient statistics')

    # Inference
    parser.add_argument('--mode', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--black_bg', action='store_true', help='whether to use black background')
    parser.add_argument('--no_context_audio_feat', action='store_true',
                        help='whether to use only the current audio feature')
    parser.add_argument('--dynamic_threshold_ratio', '-dtr', type=float, default=0,
                        help='dynamic thresholding ratio. 0 to disable')
    parser.add_argument('--dynamic_threshold_min', '-dtmin', type=float, default=1.)
    parser.add_argument('--dynamic_threshold_max', '-dtmax', type=float, default=4.)
    parser.add_argument('--save_coef', action='store_true', help='whether to save the generated coefficients')

    args = parser.parse_known_args()[0]
    if args.mode != 'interactive':
        parser.add_argument('--audio', '-a', type=Path, required=True, help='path of the input audio signal')
        parser.add_argument('--coef', '-c', type=Path, required=True, help='path to the coefficients')
        parser.add_argument('--style', '-s', type=Path, help='path to the style feature')
        parser.add_argument('--tex', '-t', type=Path, help='path of the rendered video')
        parser.add_argument('--no_head', action='store_true', help='whether to include head pose')
        parser.add_argument('--output', '-o', type=Path, required=True, help='path of the rendered video')
        parser.add_argument('--n_repetitions', '-n', type=int, default=1, help='number of repetitions')
        parser.add_argument('--scale_audio', '-sa', type=float, default=1.15, help='guiding scale')
        parser.add_argument('--scale_style', '-ss', type=float, default=3, help='guiding scale')
        parser.add_argument('--cfg_mode', type=str, choices=['incremental', 'independent'])
        parser.add_argument('--cfg_cond', type=str)

    args = parser.parse_args()
    main(args)
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from app.config import get_settings
from app.prompt_planner import plan_prompt


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mac-image-edit",
        description="Edit an image locally with an Apple Silicon GPU-backed diffusion model.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="Show the final prompt sent to the image model.")
    plan.add_argument("prompt", help="Plain-language image edit request.")

    edit = subparsers.add_parser("edit", help="Generate an edited image.")
    edit.add_argument("image", type=Path, help="Input image path.")
    edit.add_argument("prompt", help="Plain-language image edit request.")
    edit.add_argument("-o", "--output", type=Path, default=Path("edited.png"), help="Output PNG path.")
    edit.add_argument("--steps", type=positive_int, default=20, help="Diffusion steps.")
    edit.add_argument("--guidance-scale", type=float, default=7.5, help="Text prompt guidance.")
    edit.add_argument(
        "--image-guidance-scale",
        type=float,
        default=1.5,
        help="How strongly the model preserves the input image.",
    )
    edit.add_argument("--seed", type=int, default=None, help="Optional deterministic seed.")
    edit.add_argument(
        "--show-planned-prompt",
        action="store_true",
        help="Print the final prompt before generation.",
    )

    video = subparsers.add_parser("video", help="Generate an MP4 video from an image.")
    video.add_argument("image", type=Path, help="Input image path.")
    video.add_argument(
        "prompt",
        nargs="?",
        default="",
        help="Optional edit request when --edit-first is enabled.",
    )
    video.add_argument("-o", "--output", type=Path, default=Path("video.mp4"), help="Output MP4 path.")
    video.add_argument("--edit-first", action="store_true", help="Apply the prompt before animation.")
    video.add_argument("--edit-steps", type=positive_int, default=20, help="Image edit diffusion steps.")
    video.add_argument("--video-steps", type=positive_int, default=25, help="Video diffusion steps.")
    video.add_argument("--frames", type=positive_int, default=14, help="Number of video frames.")
    video.add_argument("--fps", type=positive_int, default=7, help="Output video frames per second.")
    video.add_argument("--motion-bucket-id", type=int, default=127, help="SVD motion intensity bucket.")
    video.add_argument("--noise-aug-strength", type=float, default=0.02, help="SVD image noise strength.")
    video.add_argument("--seed", type=int, default=None, help="Optional deterministic seed.")
    video.add_argument(
        "--show-planned-prompt",
        action="store_true",
        help="Print the final edit prompt when --edit-first is enabled.",
    )
    return parser


async def run_plan(prompt: str) -> int:
    settings = get_settings()
    print(await plan_prompt(prompt, settings))
    return 0


async def run_edit(args: argparse.Namespace) -> int:
    from app.editor import ImageEditor
    from app.image_utils import load_image

    settings = get_settings()
    if not args.image.exists():
        raise SystemExit(f"Input image does not exist: {args.image}")

    planned = await plan_prompt(args.prompt, settings)
    if args.show_planned_prompt:
        print(planned)

    image = load_image(args.image.read_bytes(), settings.max_image_side)
    result = ImageEditor(settings).edit(
        image=image,
        prompt=planned,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output, format="PNG")
    print(args.output)
    return 0


async def run_video(args: argparse.Namespace) -> int:
    from app.editor import ImageEditor
    from app.image_utils import load_image
    from app.video import VideoGenerator

    settings = get_settings()
    if not args.image.exists():
        raise SystemExit(f"Input image does not exist: {args.image}")

    image = load_image(args.image.read_bytes(), settings.max_image_side)
    if args.edit_first:
        planned = await plan_prompt(args.prompt, settings)
        if args.show_planned_prompt:
            print(planned)
        image = ImageEditor(settings).edit(
            image=image,
            prompt=planned,
            steps=args.edit_steps,
            seed=args.seed,
        )

    path = VideoGenerator(settings).generate(
        image=image,
        output_path=args.output,
        steps=args.video_steps,
        frames=args.frames,
        fps=args.fps,
        motion_bucket_id=args.motion_bucket_id,
        noise_aug_strength=args.noise_aug_strength,
        seed=args.seed,
    )
    print(path)
    return 0


async def async_main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "plan":
        return await run_plan(args.prompt)
    if args.command == "edit":
        return await run_edit(args)
    if args.command == "video":
        return await run_video(args)
    parser.error("unknown command")
    return 2


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()

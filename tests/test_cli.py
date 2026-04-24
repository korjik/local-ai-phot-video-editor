from app.cli import build_parser


def test_cli_parses_edit_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["edit", "input.jpg", "make it brighter", "--steps", "12"])

    assert args.command == "edit"
    assert args.prompt == "make it brighter"
    assert args.steps == 12


def test_cli_parses_plan_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["plan", "remove the background"])

    assert args.command == "plan"
    assert args.prompt == "remove the background"


def test_cli_parses_video_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["video", "input.jpg", "--frames", "12", "--fps", "6"])

    assert args.command == "video"
    assert args.frames == 12
    assert args.fps == 6

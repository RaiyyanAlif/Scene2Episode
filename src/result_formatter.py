from pathlib import Path


def format_timestamp(seconds):
    seconds = float(seconds)

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"


def confidence_label(score):
    score = float(score)

    if score >= 0.90:
        return "Very High"
    elif score >= 0.80:
        return "High"
    elif score >= 0.70:
        return "Moderate"
    else:
        return "Low"


def format_result(result, rank=1):

    candidate = result["best_candidate"]
    metadata = candidate["metadata"]

    title = metadata.get(
        "title",
        metadata.get("media_id", "Unknown"),
    )

    season = metadata.get("season")
    episode = metadata.get("episode")
    episode_title = metadata.get("episode_title")

    timestamp = format_timestamp(
        metadata.get(
            "timestamp_seconds",
            0,
        )
    )

    score = float(
        result.get(
            "score",
            candidate.get("similarity", 0),
        )
    )

    lines = []

    lines.append(
        "╔══════════════════════════════════════════════╗"
    )
    lines.append(
        "║              SCENE2EPISODE                  ║"
    )
    lines.append(
        "╠══════════════════════════════════════════════╣"
    )

    lines.append(
        f"║ Title      : {str(title):<31}║"
    )

    if season is not None:

        lines.append(
            f"║ Season     : {str(season):<31}║"
        )

    if episode is not None:

        lines.append(
            f"║ Episode    : {str(episode):<31}║"
        )

    if episode_title:

        lines.append(
            f"║ Ep. Title  : {str(episode_title)[:31]:<31}║"
        )

    lines.append(
        f"║ Timestamp  : {timestamp:<31}║"
    )

    lines.append(
        f"║ Confidence : {score * 100:>6.2f}% "
        f"({confidence_label(score)})      ║"
    )

    lines.append(
        "╚══════════════════════════════════════════════╝"
    )

    return "\n".join(lines)


def print_top_results(results):

    if not results:

        print("No results found.")

        return

    print()
    print(
        format_result(
            results[0],
            rank=1,
        )
    )

    if len(results) > 1:

        print()
        print(
            "Alternative matches:"
        )

        for rank, result in enumerate(
            results[1:],
            start=2,
        ):

            candidate = result[
                "best_candidate"
            ]

            metadata = candidate[
                "metadata"
            ]

            title = metadata.get(
                "title",
                metadata.get(
                    "media_id",
                    "Unknown",
                ),
            )

            season = metadata.get(
                "season"
            )

            episode = metadata.get(
                "episode"
            )

            timestamp = format_timestamp(
                metadata.get(
                    "timestamp_seconds",
                    0,
                )
            )

            score = float(
                result.get(
                    "score",
                    0,
                )
            )

            if (
                season is not None
                and episode is not None
            ):

                name = (
                    f"{title} "
                    f"S{int(season):02d} "
                    f"E{int(episode):02d}"
                )

            else:

                name = str(title)

            print(
                f"  {rank}. "
                f"{name:<35} "
                f"{timestamp}  "
                f"{score * 100:.2f}%"
            )
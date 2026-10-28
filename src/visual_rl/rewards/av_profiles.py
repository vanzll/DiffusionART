"""Immutable identities for the opt-in H3 three-reward recipe."""

THREE_REWARD_PROFILE = "clap_imagebind_pickscore"
THREE_REWARD_WEIGHTS = {"clap": 1., "imagebind": 1., "pickscore": 1.}
THREE_REWARD_ASSETS = frozenset({"clap", "imagebind", "imagebind_source", "pickscore", "pickscore_processor"})
PICKSCORE_REPOSITORY = "yuvalkirstain/PickScore_v1"
PICKSCORE_REVISION = "a4e4367c6dfa7288a00c550414478f865b875800"
PICKSCORE_PROCESSOR_REPOSITORY = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
PICKSCORE_PROCESSOR_REVISION = "1c2b8495b28150b8a4922ee1c8edee224c284c0c"


def validate_pickscore_provenance(provenance):
    for name, repo, revision in (
        ("pickscore", PICKSCORE_REPOSITORY, PICKSCORE_REVISION),
        ("pickscore_processor", PICKSCORE_PROCESSOR_REPOSITORY, PICKSCORE_PROCESSOR_REVISION),
    ):
        source = provenance[name]
        if source.get("repo") != repo or source.get("revision") != revision:
            raise ValueError(f"{name} must identify the pinned official repository/revision")

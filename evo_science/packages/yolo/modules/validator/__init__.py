from evo_science.packages.yolo.modules.validator.validator import Validator
from evo_science.packages.yolo.modules.validator.config import ValidatorConfig

__all__ = ["Validator", "ValidatorConfig", "validate"]


def validate(model, data_dir: str):
    """Legacy function for backward compatibility"""
    config = ValidatorConfig(
        data_dir=data_dir,
        input_size=640,
        batch_size=8,
        num_workers=4,
    )
    validator = Validator(model=model, config=config)
    return validator.validate()

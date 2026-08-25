from pathlib import Path

import hydra
import onnx
import openvino as ov
import tensorrt as trt
import torch
import torch.onnx
from loguru import logger
from omegaconf import DictConfig
from torch import nn

from src.dl.train import build_model
from src.dl.utils import get_latest_experiment_name

INPUT_NAME = "input"
OUTPUT_NAME = "output"


def load_model(model_name: str, model_path: Path, num_classes: int, device: str) -> nn.Module:
    model = build_model(
        model_name=model_name,
        pretrained=False,
        num_labels=num_classes,
        device=device,
        layers_to_train=-1,
    )
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model


def export_to_onnx(
    model: nn.Module, onnx_path: Path, x_test: torch.Tensor, max_batch_size: int, half: bool
) -> None:
    if half:
        model = model.half()
        x_test = x_test.half()
    if max_batch_size > 1:
        dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    else:
        dynamic_axes = None
    torch.onnx.export(
        model,
        x_test,
        opset_version=19,
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_axes=dynamic_axes,
        dynamo=True,
    ).save(onnx_path)

    logger.info("ONNX model exported")


def export_to_openvino(torch_model: nn.Module, ov_path: Path) -> None:
    model = ov.convert_model(input_model=torch_model)
    # rename inputs and outputs
    model.inputs[0].tensor.set_names({INPUT_NAME})
    model.outputs[0].tensor.set_names({OUTPUT_NAME})
    ov.serialize(model, str(ov_path.with_suffix(".xml")), str(ov_path.with_suffix(".bin")))
    logger.info("OpenVINO model exported")


def export_to_tensorrt(
    onnx_file_path: Path,
    half: bool,
    max_batch_size: int,
) -> None:
    tr_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(tr_logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, tr_logger)

    with open(onnx_file_path, "rb") as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return

    config = builder.create_builder_config()
    # Increase workspace memory to help with larger batch sizes
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2GB
    if half:
        config.set_flag(trt.BuilderFlag.FP16)

    if max_batch_size > 1:
        profile = builder.create_optimization_profile()
        input_tensor = network.get_input(0)
        input_name = input_tensor.name

        # Load ONNX model to get the actual input shape information
        onnx_model = onnx.load(str(onnx_file_path))

        # Find the input by name to ensure we get the correct one
        input_shape_proto = None
        for inp in onnx_model.graph.input:
            if inp.name == input_name:
                input_shape_proto = inp.type.tensor_type.shape
                break

        if input_shape_proto is None:
            raise ValueError(
                f"Could not find input '{input_name}' in ONNX model. "
                f"Available inputs: {[inp.name for inp in onnx_model.graph.input]}"
            )

        # Extract static dimensions from ONNX model
        # The first dimension (batch) should be dynamic, others should be static
        static_dims = []
        for i, dim in enumerate(input_shape_proto.dim[1:], start=1):  # Skip batch dimension
            if dim.dim_value:
                # Static dimension
                static_dims.append(int(dim.dim_value))
            elif dim.dim_param:
                # Dynamic dimension (not allowed for non-batch dims in this case)
                raise ValueError(
                    f"Cannot create TensorRT optimization profile: input shape has dynamic "
                    f"dimension at index {i} (beyond batch). Only batch dimension can be dynamic."
                )
            else:
                raise ValueError(
                    f"Cannot create TensorRT optimization profile: input shape dimension at "
                    f"index {i} is undefined."
                )

        # Set the minimum and optimal batch size to 1, and allow the maximum batch size as provided.
        min_shape = (1, *static_dims)
        opt_shape = (1, *static_dims)
        max_shape = (max_batch_size, *static_dims)

        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)

    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError(
            "Failed to build TensorRT engine. This can happen due to:\n"
            "1. Insufficient GPU memory\n"
            "2. Unsupported operations in the ONNX model\n"
            "3. Issues with dynamic batch size configuration\n"
            "Check the TensorRT logs above for more details."
        )

    with open(onnx_file_path.with_suffix(".engine"), "wb") as f:
        f.write(engine)
    logger.info("TensorRT model exported")


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg.exp = get_latest_experiment_name(cfg.exp, cfg.train.path_to_save)
    model_path = Path(cfg.train.path_to_save) / "model.pt"
    num_classes = len(cfg.train.label_to_name)

    trt_path = model_path.parent / "model.engine"
    onnx_path = model_path.parent / "model.onnx"
    ov_path = model_path.parent / "model.xml"

    model = load_model(cfg.model_name, model_path, num_classes, cfg.train.device)

    x_test = torch.randn(1, 3, *cfg.train.img_size, requires_grad=True).to(cfg.train.device)

    export_to_onnx(model, onnx_path, x_test, cfg.export.max_batch_size, cfg.export.half)
    export_to_openvino(model, ov_path)
    export_to_tensorrt(onnx_path, cfg.export.half, cfg.export.max_batch_size)

    logger.info(f"Exports saved to: {model_path.parent}")


if __name__ == "__main__":
    main()

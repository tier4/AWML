#!/usr/bin/env python3

from argparse import ArgumentParser

import git
import onnx

DOMAINS = ["3D object detection", "3D object semantic segmentation"]


def parse_args() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("onnx_path", type=str, help="Path to input ONNX model file.")
    parser.add_argument("--domain", type=str, choices=DOMAINS, help="Model domain.")
    parser.add_argument("--version", type=int, help="Model version.")
    parser.add_argument("--doc_string", type=str, help="Model description.")
    parser.add_argument("--output", type=str, help="Path to save updated ONNX file.")
    parser.add_argument("--use_hash", action="store_true", help="Use git hash as version instead of tag.")

    args = parser.parse_args()
    return args


def log_meta_info(model: onnx.ModelProto, title: str = "") -> None:
    print(
        f"{title}:"
        f"\nProducer: {model.producer_name}"
        f"\nProducer version: {model.producer_version}"
        f"\nDomain: {model.domain}"
        f"\nVersion: {model.model_version}"
        f"\nDescription: {model.doc_string}\n"
    )


def update_meta_info(
    model: onnx.ModelProto, domain: str, version: int, doc_string: str, use_hash: bool = False
) -> None:
    if domain:
        model.domain = domain
    if version:
        model.model_version = version
    if doc_string:
        model.doc_string = doc_string

    model.producer_name = "autoware-ml"
    model.producer_version = git.Repo(search_parent_directories=False).head.object.hexsha

    if use_hash:
        model.producer_version = git.Repo(search_parent_directories=False).head.object.hexsha
    else:
        model.producer_version = git.Repo(search_parent_directories=False).tags[-1].name

    log_meta_info(model, title="Updated ONNX meta info")


def save_onnx_model(model: onnx.ModelProto, output_path: str) -> None:
    onnx.save(model, output_path)

    print(f"ONNX model saved to {output_path}.\n")


def main():
    args = parse_args()

    model = onnx.load(args.onnx_path)

    log_meta_info(model, title="Input ONNX meta info")

    update_meta_info(model, args.domain, args.version, args.doc_string, args.use_hash)

    save_onnx_model(model, args.output if args.output else args.onnx_path)


if __name__ == "__main__":
    main()

from argparse import ArgumentParser

from mmpretrain import inference_model


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('model', type=str, help='Model name')
    parser.add_argument('input', type=str, help='Input image file')
    parser.add_argument('--texts', type=str, help='Input texts', default="")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    result = inference_model(
        args.model,
        args.input,
        args.texts,
    )
    print(result)


if __name__ == '__main__':
    main()

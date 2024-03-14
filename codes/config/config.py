import argparse


def generate_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--des', type=str, default='')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--model_path', type=str, default='')
    # log
    parser.add_argument('--log_path', type=str, default='./log/THUMOS14')
    # device
    parser.add_argument('--device', type=str, default='0', help='cpu or cuda-id')
    parser.add_argument('--seed', type=int, default=0, help='random seed (default: -1)')
    # dataset
    parser.add_argument('--data_path', type=str, default='./data/THUMOS14')
    parser.add_argument('--subset', type=str, default='train')
    parser.add_argument('--modality', type=str, default='both')
    parser.add_argument('--feature_fps', type=int, default=25)
    parser.add_argument('--num_classes', type=int, default=20)
    parser.add_argument('--num_worker', type=int, default=2)
    parser.add_argument('--soft_value', type=float, default=0.4)

    # model
    parser.add_argument('--feature_size', type=int, default=2048, help='size of feature (default: 2048)')
    parser.add_argument('--roi_size', type=int, default=12,
                        help='roi size for proposal features extraction (default: 12)')
    parser.add_argument('--max_proposal', type=int, default=3000,
                        help='maximum number of proposal during training (default: 1000)')
    parser.add_argument('--dropout', type=float, default=0.7)

    # optimizer
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.001)

    # train
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--num_itr', type=int, default=2000)
    parser.add_argument('--update_fre', type=int, default=200)
    parser.add_argument('--up_threshold', type=float, default=0.8)

    # test
    parser.add_argument('--refine_threshold', type=float, default=0.4)
    parser.add_argument('--gt_path', type=str, default='./data/THUMOS14/gt.json')
    parser.add_argument('--scale', type=float, default=24)

    # proposal_based
    return parser.parse_args()


if __name__ == '__main__':
    args = generate_args()
    print(vars(args))

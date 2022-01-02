from id_loss import IDLoss
from utils.data import load_img
import argparse
import os

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d0', '--dir0', help = 'Ground-truth image directory', type=str, default='./imgs/ex_dir0')
parser.add_argument('-d1', '--dir1', help = 'Generated image directory', type=str, default='./imgs/ex_dir1')
parser.add_argument('-o', '--out', help = 'Final result file', type=str, default='./ID_loss_result.txt')

# parser.add_argument('-p0','--path0', type=str, default='./imgs/ex_ref.png')
# parser.add_argument('-p1','--path1', type=str, default='./imgs/ex_p0.png')

ir_se50_weights_path = './outputs/ckpt/model_ir_se50.pth'
id_loss = IDLoss(ir_se50_weights_path)

opt = parser.parse_args()
f = open(opt.out, 'w')
files = os.listdir(opt.dir0)
result = []
for file in files:
    if os.path.exists(os.path.join(opt.dir1, file)):
        # Load images
        img_orig = load_img(os.path.join(opt.dir0, file))  # GroundTruth Image
        img_gen = load_img(os.path.join(opt.dir1, file))  # Output image

        # Compute ID loss
        dist01 = i_loss = id_loss(img_gen, img_orig)[0].detach().numpy()
        result.append(dist01)
        print('%s: %.3f' % (file, dist01))
        f.writelines('%s: %.6f\n' % (file, dist01))
f.close()
print("Average ID loss:", sum(result)/len(result))


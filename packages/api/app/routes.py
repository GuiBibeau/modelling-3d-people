import torch
import os
from app import app
from app import get_rect
from app.external.pose_estimation.modules.load_state import load_state
from app.external.pose_estimation.models.with_mobilenet import PoseEstimationWithMobileNet

@app.route('/')
@app.route('/index')
def index():
    print(os.getcwd())
    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load("./app/external/pose_estimation/checkpoint_iter_370000.pth", map_location='cpu')
    load_state(net, checkpoint)
    get_rect(net.cuda(), ['/app/sample_images/test.png'], 512)
    return "Hello World!"
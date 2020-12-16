#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
sys.path.append(os.path.dirname(__file__))
from paddle_rec_base import PaddleRecBase
import built_in
import utils

class TestPaddleRecModels(PaddleRecBase):
    """test paddle rec public models."""

    def setUp(self):
        """do something for each cases."""
        utils.cmd_shell("rm -rf logs")
        utils.cmd_shell("rm -rf increment*")
        utils.cmd_shell("rm -rf inference*")
        utils.cmd_shell("cp -r PaddleRec/models ./")

    def test_tagspace(self):
        """test contentunderstanding.tagspace."""
        self.yaml_config_name = 'models/contentunderstanding/tagspace/config.yaml'
        self.run_yaml(generate=False)
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_tagspace_gpu(self):
        """test contentunderstanding.tagspace with gpu."""
        self.yaml_config_name = "models/contentunderstanding/tagspace/config.yaml"
        sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
        utils.cmd_shell(sed_gpu_cmd)
        self.run_yaml(generate=False, cuda_devices="0")
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_dssm(self):
        """test match.dssm."""
        self.yaml_config_name = 'models/match/dssm/config.yaml'
        self.run_yaml(generate=False)
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_dssm_gpu(self):
        """test match.dssm with gpu."""
        self.yaml_config_name = "models/match/dssm/config.yaml"
        sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
        utils.cmd_shell(sed_gpu_cmd)
        self.run_yaml(generate=False, cuda_devices="0")
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_multiview_simnet(self):
        """test match.multiview_simnet."""
        self.yaml_config_name = 'models/match/multiview-simnet/config.yaml'
        self.run_yaml(generate=False)
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_multiview_simnet_gpu(self):
        """test match.multiview_simnet with gpu."""
        self.yaml_config_name = "models/match/multiview-simnet/config.yaml"
        sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
        utils.cmd_shell(sed_gpu_cmd)
        self.run_yaml(generate=False, cuda_devices="0")
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_esmm(self):
        """test multitask.esmm."""
        self.yaml_config_name = 'models/multitask/esmm/config.yaml'
        self.run_yaml(generate=False)
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_esmm_gpu(self):
        """test multitask.esmm with gpu."""
        self.yaml_config_name = "models/multitask/esmm/config.yaml"
        sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
        utils.cmd_shell(sed_gpu_cmd)
        self.run_yaml(generate=False, cuda_devices="0")
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_mmoe(self):
        """test multitask.mmoe."""
        self.yaml_config_name = 'models/multitask/mmoe/config.yaml'
        self.run_yaml(generate=False)
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_mmoe_gpu(self):
        """test multitask.mmoe with gpu."""
        self.yaml_config_name = "models/multitask/mmoe/config.yaml"
        sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
        utils.cmd_shell(sed_gpu_cmd)
        self.run_yaml(generate=False, cuda_devices="0")
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_share_bottom(self):
        """test multitask.share-bottom."""
        self.yaml_config_name = 'models/multitask/share-bottom/config.yaml'
        self.run_yaml(generate=False)
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_share_bottom_gpu(self):
        """test multitask.share-bottom with gpu."""
        self.yaml_config_name = "models/multitask/share-bottom/config.yaml"
        sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
        utils.cmd_shell(sed_gpu_cmd)
        self.run_yaml(generate=False, cuda_devices="0")
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_afm(self):
        """test ran.afm."""
        self.yaml_config_name = 'models/rank/afm/config.yaml'
        self.run_yaml(generate=False)
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_afm_gpu(self):
        """test rank.afm with gpu."""
        self.yaml_config_name = "models/rank/afm/config.yaml"
        sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
        utils.cmd_shell(sed_gpu_cmd)
        self.run_yaml(generate=False, cuda_devices="0")
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_dcn(self):
        """test ran.dcn."""
        self.yaml_config_name = 'models/rank/dcn/config.yaml'
        self.run_yaml(generate=False)
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_dcn_gpu(self):
        """test rank.afm with gpu."""
        self.yaml_config_name = "models/rank/dcn/config.yaml"
        sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
        utils.cmd_shell(sed_gpu_cmd)
        self.run_yaml(generate=False, cuda_devices="0")
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_deep_crossing(self):
        """test ran.deep_crossing."""
        self.yaml_config_name = 'models/rank/deep_crossing/config.yaml'
        self.run_yaml(generate=False)
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_deep_crossing_gpu(self):
        """test rank.deep_crossing with gpu."""
        self.yaml_config_name = "models/rank/deep_crossing/config.yaml"
        sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
        utils.cmd_shell(sed_gpu_cmd)
        self.run_yaml(generate=False, cuda_devices="0")
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_deepfm(self):
        """test ran.deepfm."""
        self.yaml_config_name = 'models/rank/deepfm/config.yaml'
        self.run_yaml(generate=False)
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_deepfm_gpu(self):
        """test rank.deepfm with gpu."""
        self.yaml_config_name = "models/rank/deepfm/config.yaml"
        sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
        utils.cmd_shell(sed_gpu_cmd)
        self.run_yaml(generate=False, cuda_devices="0")
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_din(self):
        """test ran.din."""
        self.yaml_config_name = 'models/rank/din/config.yaml'
        self.run_yaml(generate=False)
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_din_gpu(self):
        """test rank.din with gpu."""
        self.yaml_config_name = "models/rank/din/config.yaml"
        sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
        utils.cmd_shell(sed_gpu_cmd)
        self.run_yaml(generate=False, cuda_devices="0")
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_dnn(self):
        """test ran.dnn."""
        self.yaml_config_name = 'models/rank/dnn/config.yaml'
        self.run_yaml(generate=False)
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_dnn_gpu(self):
        """test rank.dnn with gpu."""
        self.yaml_config_name = "models/rank/dnn/config.yaml"
        sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
        utils.cmd_shell(sed_gpu_cmd)
        self.run_yaml(generate=False, cuda_devices="0")
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_ffm(self):
        """test ran.ffm."""
        self.yaml_config_name = 'models/rank/ffm/config.yaml'
        self.run_yaml(generate=False)
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_ffm_gpu(self):
        """test rank.ffm with gpu."""
        self.yaml_config_name = "models/rank/ffm/config.yaml"
        sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
        utils.cmd_shell(sed_gpu_cmd)
        self.run_yaml(generate=False, cuda_devices="0")
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    # 这个case 现在会hang
    # def test_fgcnn(self):
    #     """test ran.fgcnn."""
    #     self.yaml_config_name = 'models/rank/fgcnn'
    #     self.run_yaml(generate=False)
    #     built_in.equals(self.pro.returncode, 0, self.err_msg)
    #     built_in.not_contains(self.err, 'Traceback', self.err_msg)
    #
    # def test_fgcnn_gpu(self):
    #     """test rank.fgcnn with gpu."""
    #     self.yaml_config_name = "models/rank/fgcnn/config.yaml"
    #     sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
    #     utils.sed_file(sed_gpu_cmd)
    #     self.run_yaml(generate=False, cuda_devices="0")
    #     built_in.equals(self.pro.returncode, 0, self.err_msg)
    #     built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_fm(self):
        """test ran.fm."""
        self.yaml_config_name = 'models/rank/fm/config.yaml'
        self.run_yaml(generate=False)
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_fm_gpu(self):
        """test rank.fm with gpu."""
        self.yaml_config_name = "models/rank/fm/config.yaml"
        sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
        utils.cmd_shell(sed_gpu_cmd)
        self.run_yaml(generate=False, cuda_devices="0")
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_fnn(self):
        """test ran.fnn."""
        self.yaml_config_name = 'models/rank/fnn/config.yaml'
        self.run_yaml(generate=False)
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_fnn_gpu(self):
        """test rank.fnn with gpu."""
        self.yaml_config_name = "models/rank/fnn/config.yaml"
        sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
        utils.cmd_shell(sed_gpu_cmd)
        self.run_yaml(generate=False, cuda_devices="0")
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_logistic_regression(self):
        """test ran.logistic_regression."""
        self.yaml_config_name = 'models/rank/logistic_regression/config.yaml'
        self.run_yaml(generate=False)
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_logistic_regression_gpu(self):
        """test rank.logistic_regression with gpu."""
        self.yaml_config_name = "models/rank/logistic_regression/config.yaml"
        sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
        utils.cmd_shell(sed_gpu_cmd)
        self.run_yaml(generate=False, cuda_devices="0")
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_nfm(self):
        """test ran.nfm."""
        self.yaml_config_name = 'models/rank/nfm/config.yaml'
        self.run_yaml(generate=False)
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_nfm_gpu(self):
        """test rank.nfm with gpu."""
        self.yaml_config_name = "models/rank/nfm/config.yaml"
        sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
        utils.cmd_shell(sed_gpu_cmd)
        self.run_yaml(generate=False, cuda_devices="0")
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    # 需要paddle>=2.0
    # def test_pnn(self):
    #     """test ran.pnn."""
    #     self.yaml_config_name = 'PaddleRec.models.rank.pnn'
    #     self.run_yaml(generate=False)
    #     built_in.equals(self.pro.returncode, 0, self.err_msg)
    #     built_in.not_contains(self.err, 'Traceback', self.err_msg)
    #
    # def test_pnn_gpu(self):
    #     """test rank.pnn with gpu."""
    #     self.yaml_config_name = "models/rank/pnn/config.yaml"
    #     sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
    #     utils.cmd_shell(sed_gpu_cmd)
    #     self.run_yaml(generate=False, cuda_devices="0")
    #     built_in.equals(self.pro.returncode, 0, self.err_msg)
    #     built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_wide_deep(self):
        """test ran.wide_deep."""
        self.yaml_config_name = 'models/rank/wide_deep/config.yaml'
        self.run_yaml(generate=False)
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_wide_deep_gpu(self):
        """test rank.wide_deep with gpu."""
        self.yaml_config_name = "models/rank/wide_deep/config.yaml"
        sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
        utils.cmd_shell(sed_gpu_cmd)
        self.run_yaml(generate=False, cuda_devices="0")
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_xdeepfm(self):
        """test ran.xdeepfm."""
        self.yaml_config_name = 'models/rank/xdeepfm/config.yaml'
        self.run_yaml(generate=False)
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_xdeepfm_gpu(self):
        """test rank.xdeepfm with gpu."""
        self.yaml_config_name = "models/rank/xdeepfm/config.yaml"
        sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
        utils.cmd_shell(sed_gpu_cmd)
        self.run_yaml(generate=False, cuda_devices="0")
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

   # def test_fibinet(self):
   #     """test rank.fibinet"""
   #     self.yaml_config_name = "models/rank/fibinet/config.yaml"
   #     verify_epochs = "sed -i 's/epochs:.*4/epochs: 1/g' {}".format(self.yaml_config_name)
   #     utils.cmd_shell(verify_epochs)
   #     self.run_yaml(generate=False)
   #     built_in.equals(self.pro.returncode, 0, self.err_msg)
   #     built_in.not_contains(self.err, 'Traceback', self.err_msg)

   # def test_fibinet_gpu(self):
   #     """test rank.fibinet"""
   #     self.yaml_config_name = "models/rank/fibinet/config.yaml"
   #     verify_epochs = "sed -i 's/epochs:.*4/epochs: 1/g' {}".format(self.yaml_config_name)
   #     sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
   #     utils.cmd_shell(sed_gpu_cmd)
   #     utils.cmd_shell(verify_epochs)
   #     self.run_yaml(generate=False, cuda_devices="0")
   #     built_in.equals(self.pro.returncode, 0, self.err_msg)
   #     built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_AutoInt(self):
        """test rank.fibinet"""
        self.yaml_config_name = "models/rank/AutoInt/config.yaml"
        verify_epochs = "sed -i 's/epochs:.*2/epochs: 1/g' {}".format(self.yaml_config_name)
        utils.cmd_shell(verify_epochs)
        self.run_yaml(generate=False)
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_AutoInt_gpu(self):
        """test rank.fibinet"""
        self.yaml_config_name = "models/rank/AutoInt/config.yaml"
        verify_epochs = "sed -i 's/epochs:.*2/epochs: 1/g' {}".format(self.yaml_config_name)
        sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
        utils.cmd_shell(sed_gpu_cmd)
        utils.cmd_shell(verify_epochs)
        self.run_yaml(generate=False, cuda_devices="0")
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_fasttext(self):
        """test recall.fasttext."""
        self.yaml_config_name = 'models/recall/fasttext/config.yaml'
        self.run_yaml(generate=False)
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_fasttext_gpu(self):
        """test recall.fasttext with gpu."""
        self.yaml_config_name = "models/recall/fasttext/config.yaml"
        sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
        utils.cmd_shell(sed_gpu_cmd)
        self.run_yaml(generate=False, cuda_devices="0")
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_gnn(self):
        """test recall.gnn."""
        self.yaml_config_name = 'models/recall/gnn/config.yaml'
        self.run_yaml(generate=False)
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_gnn_gpu(self):
        """test recall.gnn with gpu."""
        self.yaml_config_name = "models/recall/gnn/config.yaml"
        sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
        utils.cmd_shell(sed_gpu_cmd)
        self.run_yaml(generate=False, cuda_devices="0")
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

#     def test_gru4rec(self):
#        """test recall.gru4rec."""
#        self.yaml_config_name = 'models/recall/gru4rec/config.yaml'
#        self.run_yaml(generate=False)
#        built_in.equals(self.pro.returncode, 0, self.err_msg)
#        built_in.not_contains(self.err, 'Traceback', self.err_msg)

#    def test_gru4rec_gpu(self):
#        """test recall.gru4rec with gpu."""
#        self.yaml_config_name = "models/recall/gru4rec/config.yaml"
#        sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
#         utils.cmd_shell(sed_gpu_cmd)
#         self.run_yaml(generate=False, cuda_devices="0")
#         built_in.equals(self.pro.returncode, 0, self.err_msg)
#         built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_ncf(self):
        """test recall.ncf."""
        self.yaml_config_name = 'models/recall/ncf/config.yaml'
        self.run_yaml(generate=False)
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

#     def test_ncf_gpu(self):
#         """test recall.ncf with gpu."""
#         self.yaml_config_name = "models/recall/ncf/config.yaml"
#         sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
#         utils.cmd_shell(sed_gpu_cmd)
#         self.run_yaml(generate=False, cuda_devices="0")
#         built_in.equals(self.pro.returncode, 0, self.err_msg)
#         built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_ssr(self):
        """test recall.ssr."""
        self.yaml_config_name = 'models/recall/ssr/config.yaml'
        self.run_yaml(generate=False)
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_ssr_gpu(self):
        """test recall.ssr with gpu."""
        self.yaml_config_name = "models/recall/ssr/config.yaml"
        sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
        utils.cmd_shell(sed_gpu_cmd)
        self.run_yaml(generate=False, cuda_devices="0")
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_word2vec(self):
        """test recall.word2vec."""
        self.yaml_config_name = 'models/recall/word2vec/config.yaml'
        self.run_yaml(generate=False)
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_word2vec_gpu(self):
        """test recall.word2vec with gpu."""
        self.yaml_config_name = "models/recall/word2vec/config.yaml"
        sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
        utils.cmd_shell(sed_gpu_cmd)
        self.run_yaml(generate=False, cuda_devices="0")
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_youtube_dnn(self):
        """test recall.youtube_dnn."""
        self.yaml_config_name = 'models/recall/youtube_dnn/config.yaml'
        self.run_yaml(generate=False)
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_youtube_dnn_gpu(self):
        """test recall.youtube_dnn with gpu."""
        self.yaml_config_name = "models/recall/youtube_dnn/config.yaml"
        sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
        utils.cmd_shell(sed_gpu_cmd)
        self.run_yaml(generate=False, cuda_devices="0")
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_listwise(self):
        """test rerank.listwise."""
        self.yaml_config_name = 'models/rerank/listwise/config.yaml'
        self.run_yaml(generate=False)
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_listwise_gpu(self):
        """test rerank.listwise with gpu."""
        self.yaml_config_name = "models/rerank/listwise/config.yaml"
        sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
        utils.cmd_shell(sed_gpu_cmd)
        self.run_yaml(generate=False, cuda_devices="0")
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_tdm(self):
        """test treebased.tdm."""
        self.yaml_config_name = 'models/treebased/tdm/config.yaml'
        self.run_yaml(generate=False)
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    # 不支持gpu，需要paddle版本>=1.8.0
    # def test_tdm_gpu(self):
    #     """test treebased.tdm with gpu."""
    #     self.yaml_config_name = "models/treebased/tdm/config.yaml"
    #     sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
    #     utils.cmd_shell(sed_gpu_cmd)
    #     self.run_yaml(generate=False, cuda_devices="0")
    #     built_in.equals(self.pro.returncode, 0, self.err_msg)
    #     built_in.not_contains(self.err, 'Traceback', self.err_msg)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
#======================================================================
#
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
#
#======================================================================

"""
@Desc: test_paddle_models module
@File: test_paddle_models.py
@Author: liangjinhua
@Date: 2020/06/10 16:16
"""
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

    def test_tagspace(self):
        """test contentunderstanding.tagspace."""
        self.yaml_config_name = 'paddlerec.models.contentunderstanding.tagspace'
        self.run_yaml(generate=False)
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_tagspace_gpu(self):
        """test contentunderstanding.tagspace with gpu."""
        self.yaml_config_name = "./PaddleRec/models/contentunderstanding/tagspace/config.yaml"
        sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
        utils.cmd_shell(sed_gpu_cmd)
        self.run_yaml(generate=False, cuda_devices="0")
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_classification(self):
        """test contentunderstanding.classification."""
        self.yaml_config_name = 'paddlerec.models.contentunderstanding.classification'
        self.run_yaml(generate=False)
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_classification_gpu(self):
        """test contentunderstanding.classification with gpu."""
        self.yaml_config_name = "./PaddleRec/models/contentunderstanding/classification/config.yaml"
        sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
        utils.cmd_shell(sed_gpu_cmd)
        self.run_yaml(generate=False, cuda_devices="0")
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_dssm(self):
        """test match.dssm."""
        self.yaml_config_name = 'paddlerec.models.match.dssm'
        self.run_yaml(generate=False)
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_dssm_gpu(self):
        """test match.dssm with gpu."""
        self.yaml_config_name = "./PaddleRec/models/match/dssm/config.yaml"
        sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
        utils.cmd_shell(sed_gpu_cmd)
        self.run_yaml(generate=False, cuda_devices="0")
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_multiview_simnet(self):
        """test match.multiview_simnet."""
        self.yaml_config_name = 'paddlerec.models.match.multiview-simnet'
        self.run_yaml(generate=False)
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_multiview_simnet_gpu(self):
        """test match.multiview_simnet with gpu."""
        self.yaml_config_name = "./PaddleRec/models/match/multiview-simnet/config.yaml"
        sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
        utils.cmd_shell(sed_gpu_cmd)
        self.run_yaml(generate=False, cuda_devices="0")
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_esmm(self):
        """test multitask.esmm."""
        self.yaml_config_name = 'paddlerec.models.multitask.esmm'
        self.run_yaml(generate=False)
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_esmm_gpu(self):
        """test multitask.esmm with gpu."""
        self.yaml_config_name = "./PaddleRec/models/multitask/esmm/config.yaml"
        sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
        utils.cmd_shell(sed_gpu_cmd)
        self.run_yaml(generate=False, cuda_devices="0")
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_mmoe(self):
        """test multitask.mmoe."""
        self.yaml_config_name = 'paddlerec.models.multitask.mmoe'
        self.run_yaml(generate=False)
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_mmoe_gpu(self):
        """test multitask.mmoe with gpu."""
        self.yaml_config_name = "./PaddleRec/models/multitask/mmoe/config.yaml"
        sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
        utils.cmd_shell(sed_gpu_cmd)
        self.run_yaml(generate=False, cuda_devices="0")
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_share_bottom(self):
        """test multitask.share-bottom."""
        self.yaml_config_name = 'paddlerec.models.multitask.share-bottom'
        self.run_yaml(generate=False)
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_share_bottom_gpu(self):
        """test multitask.share-bottom with gpu."""
        self.yaml_config_name = "./PaddleRec/models/multitask/share-bottom/config.yaml"
        sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
        utils.cmd_shell(sed_gpu_cmd)
        self.run_yaml(generate=False, cuda_devices="0")
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_afm(self):
        """test ran.afm."""
        self.yaml_config_name = 'paddlerec.models.rank.afm'
        self.run_yaml(generate=False)
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_afm_gpu(self):
        """test rank.afm with gpu."""
        self.yaml_config_name = "./PaddleRec/models/rank/afm/config.yaml"
        sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
        utils.cmd_shell(sed_gpu_cmd)
        self.run_yaml(generate=False, cuda_devices="0")
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_dcn(self):
        """test ran.dcn."""
        self.yaml_config_name = 'paddlerec.models.rank.dcn'
        self.run_yaml(generate=False)
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_dcn_gpu(self):
        """test rank.afm with gpu."""
        self.yaml_config_name = "./PaddleRec/models/rank/dcn/config.yaml"
        sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
        utils.cmd_shell(sed_gpu_cmd)
        self.run_yaml(generate=False, cuda_devices="0")
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_deep_crossing(self):
        """test ran.deep_crossing."""
        self.yaml_config_name = 'paddlerec.models.rank.deep_crossing'
        self.run_yaml(generate=False)
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_deep_crossing_gpu(self):
        """test rank.deep_crossing with gpu."""
        self.yaml_config_name = "./PaddleRec/models/rank/deep_crossing/config.yaml"
        sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
        utils.cmd_shell(sed_gpu_cmd)
        self.run_yaml(generate=False, cuda_devices="0")
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_deepfm(self):
        """test ran.deepfm."""
        self.yaml_config_name = 'paddlerec.models.rank.deepfm'
        self.run_yaml(generate=False)
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_deepfm_gpu(self):
        """test rank.deepfm with gpu."""
        self.yaml_config_name = "./PaddleRec/models/rank/deepfm/config.yaml"
        sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
        utils.cmd_shell(sed_gpu_cmd)
        self.run_yaml(generate=False, cuda_devices="0")
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_din(self):
        """test ran.din."""
        self.yaml_config_name = 'paddlerec.models.rank.din'
        self.run_yaml(generate=False)
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_din_gpu(self):
        """test rank.din with gpu."""
        self.yaml_config_name = "./PaddleRec/models/rank/din/config.yaml"
        sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
        utils.cmd_shell(sed_gpu_cmd)
        self.run_yaml(generate=False, cuda_devices="0")
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_dnn(self):
        """test ran.dnn."""
        self.yaml_config_name = 'paddlerec.models.rank.dnn'
        self.run_yaml(generate=False)
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_dnn_gpu(self):
        """test rank.dnn with gpu."""
        self.yaml_config_name = "./PaddleRec/models/rank/dnn/config.yaml"
        sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
        utils.cmd_shell(sed_gpu_cmd)
        self.run_yaml(generate=False, cuda_devices="0")
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_ffm(self):
        """test ran.ffm."""
        self.yaml_config_name = 'paddlerec.models.rank.ffm'
        self.run_yaml(generate=False)
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_ffm_gpu(self):
        """test rank.ffm with gpu."""
        self.yaml_config_name = "./PaddleRec/models/rank/ffm/config.yaml"
        sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
        utils.cmd_shell(sed_gpu_cmd)
        self.run_yaml(generate=False, cuda_devices="0")
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    # 这个case 现在会hang
    # def test_fgcnn(self):
    #     """test ran.fgcnn."""
    #     self.yaml_config_name = 'paddlerec.models.rank.fgcnn'
    #     self.run_yaml(generate=False)
    #     built_in.equals(self.pro.returncode, 0, self.err_msg)
    #     built_in.not_contains(self.err, 'Traceback', self.err_msg)
    #
    # def test_fgcnn_gpu(self):
    #     """test rank.fgcnn with gpu."""
    #     self.yaml_config_name = "./PaddleRec/models/rank/fgcnn/config.yaml"
    #     sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
    #     utils.sed_file(sed_gpu_cmd)
    #     self.run_yaml(generate=False, cuda_devices="0")
    #     built_in.equals(self.pro.returncode, 0, self.err_msg)
    #     built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_fm(self):
        """test ran.fm."""
        self.yaml_config_name = 'paddlerec.models.rank.fm'
        self.run_yaml(generate=False)
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_fm_gpu(self):
        """test rank.fm with gpu."""
        self.yaml_config_name = "./PaddleRec/models/rank/fm/config.yaml"
        sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
        utils.cmd_shell(sed_gpu_cmd)
        self.run_yaml(generate=False, cuda_devices="0")
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_fnn(self):
        """test ran.fnn."""
        self.yaml_config_name = 'paddlerec.models.rank.fnn'
        self.run_yaml(generate=False)
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_fnn_gpu(self):
        """test rank.fnn with gpu."""
        self.yaml_config_name = "./PaddleRec/models/rank/fnn/config.yaml"
        sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
        utils.cmd_shell(sed_gpu_cmd)
        self.run_yaml(generate=False, cuda_devices="0")
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_logistic_regression(self):
        """test ran.logistic_regression."""
        self.yaml_config_name = 'paddlerec.models.rank.logistic_regression'
        self.run_yaml(generate=False)
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_logistic_regression_gpu(self):
        """test rank.logistic_regression with gpu."""
        self.yaml_config_name = "./PaddleRec/models/rank/logistic_regression/config.yaml"
        sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
        utils.cmd_shell(sed_gpu_cmd)
        self.run_yaml(generate=False, cuda_devices="0")
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_nfm(self):
        """test ran.nfm."""
        self.yaml_config_name = 'paddlerec.models.rank.nfm'
        self.run_yaml(generate=False)
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_nfm_gpu(self):
        """test rank.nfm with gpu."""
        self.yaml_config_name = "./PaddleRec/models/rank/nfm/config.yaml"
        sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
        utils.cmd_shell(sed_gpu_cmd)
        self.run_yaml(generate=False, cuda_devices="0")
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    # 需要paddle>=2.0
    # def test_pnn(self):
    #     """test ran.pnn."""
    #     self.yaml_config_name = 'paddlerec.models.rank.pnn'
    #     self.run_yaml(generate=False)
    #     built_in.equals(self.pro.returncode, 0, self.err_msg)
    #     built_in.not_contains(self.err, 'Traceback', self.err_msg)
    #
    # def test_pnn_gpu(self):
    #     """test rank.pnn with gpu."""
    #     self.yaml_config_name = "./PaddleRec/models/rank/pnn/config.yaml"
    #     sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
    #     utils.cmd_shell(sed_gpu_cmd)
    #     self.run_yaml(generate=False, cuda_devices="0")
    #     built_in.equals(self.pro.returncode, 0, self.err_msg)
    #     built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_wide_deep(self):
        """test ran.wide_deep."""
        self.yaml_config_name = 'paddlerec.models.rank.wide_deep'
        self.run_yaml(generate=False)
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_wide_deep_gpu(self):
        """test rank.wide_deep with gpu."""
        self.yaml_config_name = "./PaddleRec/models/rank/wide_deep/config.yaml"
        sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
        utils.cmd_shell(sed_gpu_cmd)
        self.run_yaml(generate=False, cuda_devices="0")
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_xdeepfm(self):
        """test ran.xdeepfm."""
        self.yaml_config_name = 'paddlerec.models.rank.xdeepfm'
        self.run_yaml(generate=False)
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_xdeepfm_gpu(self):
        """test rank.xdeepfm with gpu."""
        self.yaml_config_name = "./PaddleRec/models/rank/xdeepfm/config.yaml"
        sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
        utils.cmd_shell(sed_gpu_cmd)
        self.run_yaml(generate=False, cuda_devices="0")
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_fasttext(self):
        """test recall.fasttext."""
        self.yaml_config_name = 'paddlerec.models.recall.fasttext'
        self.run_yaml(generate=False)
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_fasttext_gpu(self):
        """test recall.fasttext with gpu."""
        self.yaml_config_name = "./PaddleRec/models/recall/fasttext/config.yaml"
        sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
        utils.cmd_shell(sed_gpu_cmd)
        self.run_yaml(generate=False, cuda_devices="0")
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_gnn(self):
        """test recall.gnn."""
        self.yaml_config_name = 'paddlerec.models.recall.gnn'
        self.run_yaml(generate=False)
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_gnn_gpu(self):
        """test recall.gnn with gpu."""
        self.yaml_config_name = "./PaddleRec/models/recall/gnn/config.yaml"
        sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
        utils.cmd_shell(sed_gpu_cmd)
        self.run_yaml(generate=False, cuda_devices="0")
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_gru4rec(self):
        """test recall.gru4rec."""
        self.yaml_config_name = 'paddlerec.models.recall.gru4rec'
        self.run_yaml(generate=False)
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_gru4rec_gpu(self):
        """test recall.gru4rec with gpu."""
        self.yaml_config_name = "./PaddleRec/models/recall/gru4rec/config.yaml"
        sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
        utils.cmd_shell(sed_gpu_cmd)
        self.run_yaml(generate=False, cuda_devices="0")
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_ncf(self):
        """test recall.ncf."""
        self.yaml_config_name = 'paddlerec.models.recall.ncf'
        self.run_yaml(generate=False)
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_ncf_gpu(self):
        """test recall.ncf with gpu."""
        self.yaml_config_name = "./PaddleRec/models/recall/ncf/config.yaml"
        sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
        utils.cmd_shell(sed_gpu_cmd)
        self.run_yaml(generate=False, cuda_devices="0")
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_ssr(self):
        """test recall.ssr."""
        self.yaml_config_name = 'paddlerec.models.recall.ssr'
        self.run_yaml(generate=False)
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_ssr_gpu(self):
        """test recall.ssr with gpu."""
        self.yaml_config_name = "./PaddleRec/models/recall/ssr/config.yaml"
        sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
        utils.cmd_shell(sed_gpu_cmd)
        self.run_yaml(generate=False, cuda_devices="0")
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_word2vec(self):
        """test recall.word2vec."""
        self.yaml_config_name = 'paddlerec.models.recall.word2vec'
        self.run_yaml(generate=False)
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_word2vec_gpu(self):
        """test recall.word2vec with gpu."""
        self.yaml_config_name = "./PaddleRec/models/recall/word2vec/config.yaml"
        sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
        utils.cmd_shell(sed_gpu_cmd)
        self.run_yaml(generate=False, cuda_devices="0")
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_youtube_dnn(self):
        """test recall.youtube_dnn."""
        self.yaml_config_name = 'paddlerec.models.recall.youtube_dnn'
        self.run_yaml(generate=False)
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_youtube_dnn_gpu(self):
        """test recall.youtube_dnn with gpu."""
        self.yaml_config_name = "./PaddleRec/models/recall/youtube_dnn/config.yaml"
        sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
        utils.cmd_shell(sed_gpu_cmd)
        self.run_yaml(generate=False, cuda_devices="0")
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_listwise(self):
        """test rerank.listwise."""
        self.yaml_config_name = 'paddlerec.models.rerank.listwise'
        self.run_yaml(generate=False)
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    def test_listwise_gpu(self):
        """test rerank.listwise with gpu."""
        self.yaml_config_name = "./PaddleRec/models/rerank/listwise/config.yaml"
        sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
        utils.cmd_shell(sed_gpu_cmd)
        self.run_yaml(generate=False, cuda_devices="0")
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)


    def test_tdm(self):
        """test treebased.tdm."""
        self.yaml_config_name = 'paddlerec.models.treebased.tdm'
        self.run_yaml(generate=False)
        built_in.equals(self.pro.returncode, 0, self.err_msg)
        built_in.not_contains(self.err, 'Traceback', self.err_msg)

    # 不支持gpu，需要paddle版本>=1.8.0
    # def test_tdm_gpu(self):
    #     """test treebased.tdm with gpu."""
    #     self.yaml_config_name = "./PaddleRec/models/treebased/tdm/config.yaml"
    #     sed_gpu_cmd = "sed -i 's/device:.*cpu/device: gpu/g' {}".format(self.yaml_config_name)
    #     utils.cmd_shell(sed_gpu_cmd)
    #     self.run_yaml(generate=False, cuda_devices="0")
    #     built_in.equals(self.pro.returncode, 0, self.err_msg)
    #     built_in.not_contains(self.err, 'Traceback', self.err_msg)


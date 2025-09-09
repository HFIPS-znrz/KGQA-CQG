#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import division, unicode_literals
import argparse

from onmt.utils.logging import init_logger
from onmt.translate.translator import build_translator

import onmt.inputters
import onmt.translate
import onmt
import onmt.model_builder
import onmt.modules
import onmt.opts

import sys
import merge

class NullWriter:
    def write(self, _):
        pass
    def flush(self):
        pass

# 重定向 sys.stdout
sys.stdout = NullWriter()

def main(opt):
    translator,translator1, translator2 = build_translator(opt, report_score=True)
    #print(translator)
    scores1, predictions1, dec_out1 = translator1.translate(src_path=opt.src,
                         tgt_path=opt.tgt,
                         src_dir=opt.src_dir,
                         batch_size=opt.batch_size,
                         attn_debug=opt.attn_debug)
    scores2, predictions2, dec_out2 = translator2.translate(src_path=opt.src,
                         tgt_path=opt.tgt,
                         src_dir=opt.src_dir,
                         batch_size=opt.batch_size,
                         attn_debug=opt.attn_debug)
    translator.translate(src_path=opt.src,
                         tgt_path=opt.tgt,
                         src_dir=opt.src_dir,
                         batch_size=opt.batch_size,
                         attn_debug=opt.attn_debug,
                         dec_out1=dec_out1,
                         dec_out2=dec_out2)
    
    


def test(category_name, test_data, rel, know1, know2):
    with open('/home/dell/zjx/MSParS_V2.0_single_2_newme/code/generation/OpenNMT/data.txt', 'w') as f:
        content = test_data + ' <SP> ' + category_name + ' <S> ' + rel +' <S> '+ know1 +' <SP> ' +  category_name + ' <S> '+rel +' <S> '+know2
        f.write(content)
        f.write('\n')
        f.write(content)
    # 桂单0811什么时候播种？ <SP> 桂单0811 <S> 播期 <S> 春播宜在当地气温稳定在10℃以上，土壤湿度为60%左右时播种。 <SP> 桂单0811 <S> 播期 <S> 秋播宜在立秋前后种植。
    
    parser = argparse.ArgumentParser(
        description='translate.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    onmt.opts.add_md_help_argument(parser)
    onmt.opts.translate_opts(parser)

    opt = parser.parse_args()
    logger = init_logger(opt.log_file)
    main(opt)
    # clari = merge.mer()
    # return clari
    

if __name__ == '__main__':
    # test("鼎玉678","鼎玉678的施肥方法是什么？","施肥","基肥施N、P、K复合肥20-30千克/亩","拔节期肥拔节期追施尿素40-50千克/亩")
    # test("邯玉66", "邯玉66适宜的种植密度是多少？", "种植密度", "一般地块种植密度4200株/亩", "高产地块密度4500株/亩。")
    # test("斯达糯54", "斯达糯54什么时候播种？", "播期", "北方鲜食玉米适宜播种期4月中旬至6月中旬", "黄淮海鲜食玉米类型区适宜播种期4月初至7月中旬")
    test(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    # print(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    # sys.stdout = sys.__stdout__
    # print(clari)
    # print(knowledge)
    # knowledge = test('玉农76','玉农76适宜的种植密度是多少？')
    # sys.exit(' '.join(knowledge))


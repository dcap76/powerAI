# coding: utf8

import os

import jieba

BASE_DIR = r'traindata'

for root, dirnames, filenames in os.walk(BASE_DIR):
    rows = []
    train_fenlei_filename = 'train_feilei.csv'
    train_qinggan_filename = 'train_qinggan.csv'
    with open(train_fenlei_filename, mode='wb+') as f, open(train_qinggan_filename, mode='wb+') as tqf:
        # HEAD
        f.write('Category|Descript\n')
        tqf.write('Category|Descript\n')
        for filename in filenames:
            content = ''
            label = ''
            label_filename = '%s-1.txt' % filename.replace(r'.txt', '')
            qinggan_filename = '%s-2.txt' % filename.replace(r'.txt', '')

            # print label_filename
            if os.path.isfile(os.path.join(root, label_filename)):
                with open(os.path.join(root, filename)) as fn, open(os.path.join(root, label_filename)) as lf, open(
                        os.path.join(root, qinggan_filename)) as qgf:
                    content = fn.read().replace(r'\n', '')
                    label = lf.read()
                    qinggan = qgf.read()

                    # fenlei
                    row = '{}|{}\n'.format(label, ' '.join(jieba.lcut(content)).encode('utf-8'))
                    f.write(row)

                    # qinggan
                    ganqing_row = '{}|{}\n'.format(qinggan, ' '.join(jieba.lcut(content)).encode('utf-8'))
                    tqf.write(ganqing_row)

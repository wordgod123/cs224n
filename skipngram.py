#!python
#coding=utf8

import tensorflow as tf
import jieba
import chardet
from chardet.universaldetector import UniversalDetector
import math
import collections
import numpy as np
import random
from six.moves import xrange
import os
from tensorflow.contrib.tensorboard.plugins import projector
import platform
import xmldeal
import sys
import json
from threading import Thread, Lock
import queue

# tf.enable_eager_execution()

data_index = 0
DATA_QUEUE = queue.Queue()
DATA_LOCK = Lock()

def main():
    pass

def get_coding(filename):
    detector = UniversalDetector()
    detector.reset()
    for line in open(filename, "rb"):
        detector.feed(line)
        if detector.done: break
    detector.close()
    result = detector.result
    # print(result)
    return result["encoding"]

def split_words(filename, encoding="utf-8"):
    _content_lists = []
    with open(filename, "r", encoding=encoding) as f:
        c = f.read()
        for line in c.split("\n"):
            if not line.strip():
                continue
            _content_lists.extend(list(jieba.cut(line.strip().replace(" ",""), cut_all=False)))
    return _content_lists

def split_words_with_content(content):
    _content_lists = []
    for line in content.split("\n"):
        if not line.strip():
            continue
        _content_lists.extend(list(jieba.cut(line.strip().replace(" ",""), cut_all=False)))
    return _content_lists

def build_dataset(words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    counter = collections.Counter(words)
    n_words = len(counter)
    #统计每个单词的频率
    count.extend(counter.most_common(n_words))
    vocabulary_size = n_words + 1

    #单词 one_hoting 字典
    dictionary = dict()
    for word, _ in count:
      dictionary[word] = len(dictionary)

    #one_hoting后的数据，将words中词汇的one_hoting的index写入data列表中
    data = list()
    unk_count = 0
    for word in words:
      index = dictionary.get(word, 0)
      if index == 0:  # dictionary['UNK']和不在dictionary的词汇都将统一放到UNK词汇中
        unk_count += 1
      data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary, vocabulary_size

def generate_batch(batch_size, num_skips, skip_window):
  '''
        @note: num_skips: 使用skip_window内的多少个词汇进行训练，比如skip_window为2，num_skips为3, 则表示在左右共4个词汇中选出3个作为目标词汇进行训练
        @note: skip_window: source左右多少个词汇作为目标进行训练
  '''
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

  #滑动窗口大小
  span = 2 * skip_window + 1  # [ skip_window target skip_window ]

  #生成一个最大长度的类似于列表的东西，但是只会保留span个数据，当大于span个数据，则会将最开始的数据丢弃掉保留新的数据
  buffer = collections.deque(maxlen=span)  # pylint: disable=redefined-builtin

  #当最大index大于了数据总长度，则将data_index归0从头开始
  if data_index + span > len(data):
    data_index = 0

  #窗口中的数据
  buffer.extend(data[data_index:data_index + span])
  data_index += span

  #除以num_skips使总的生成的大小为batch_size
  for i in range(batch_size // num_skips):
    #得到目标的数据列表
    context_words = [w for w in range(span) if w != skip_window]
    #使用随机数获得最终用于作为目标数据的列表
    words_to_use = random.sample(context_words, num_skips)
    #将源、目标分别写入batch和labels列表中
    for j, context_word in enumerate(words_to_use):
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[context_word]

    #当data_index已经是最后一个的时候，则从位置0开始重新添加数据到buffer队列
    if data_index == len(data):
      buffer.extend(data[0:span])
      data_index = span
    #否则向左移1位进行下一个源的目标生成
    else:
      buffer.append(data[data_index])
      data_index += 1
  # Backtrack a little bit to avoid skipping words in the end of a batch
  data_index = (data_index + len(data) - span) % len(data)
  return batch, labels

def get_sdmz():
    if platform.platform().lower().find("windows")>=0:
        files = "D:\\eclipse\\cs224n\\own\\sdmz\\红楼梦全集txt\\红楼梦01.txt"
    else:
        files = "/home/cc/data/sdmz/红楼梦全集txt/红楼梦01.txt"
        
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)

    encoding = get_coding(files)
    content_lists = split_words(files, encoding)
    return content_lists

def _sogou_get_content(dirpath):
    global content_lists
    x = xmldeal.xmlParse()
    while DATA_QUEUE.qsize() > 0:
        filename = DATA_QUEUE.get()
        x.parseXmlFromFile(dirpath + "/" + filename)
        news_content = x.getContent("content")
        if news_content ==None or news_content.strip() == "":
            continue
        vocabs = split_words_with_content(news_content)
        if DATA_LOCK.acquire():
            content_lists.extend(vocabs)
            DATA_LOCK.release()

def get_sogou_news(dirpath=None):
    if dirpath is None:
        dirpath = "/home/cc/data/news_sohusite_xml.full/split/"
    filelist = os.listdir(dirpath)
    
    READ_FILE_NUMBER = 50000
    # content_lists = []
    x = xmldeal.xmlParse()
    for i in range(READ_FILE_NUMBER):
        if i < len(filelist):
            DATA_QUEUE.put(filelist[i])
            # # print(dirpath + "/" + filelist[i])
            # x.parseXmlFromFile(dirpath + "/" + filelist[i])
            # news_content = x.getContent("content")
            # # print(news_content)
            # if news_content ==None or news_content.strip() == "":
            #     continue
            # content_lists.extend(split_words_with_content(news_content))
        else:
            break
    #生成线程处理数据
    THREAD_NUM = 8
    THREAD_LIST = []
    for i in range(THREAD_NUM):
        t = Thread(target=_sogou_get_content, args=(dirpath,))
        t.start()
        THREAD_LIST.append(t)
    for i in THREAD_LIST:
        i.join()
    if DATA_QUEUE.qsize() > 0:
        print("parse xmls failed with some error")
        sys.exit(1)
    else:
        print("parse xml success")
    # return content_lists

if __name__ == "__main__":
    embedding_size = 300  # Dimension of the embedding vector.
    batch_size = 128 
    num_skips = 4
    skip_window = 2
    num_sampled = 64
    num_steps = 500000

    dataset = "sogou"

    TYPE_LIST = ["train", "predict"]
    if len(sys.argv) == 1:
        TYPE = "train"
    else:
        TYPE = sys.argv[1]
    if TYPE not in TYPE_LIST:
        print("%s not in [%s]"%(TYPE, "|".join(TYPE_LIST)))

    # print(platform.platform())
    if platform.platform().lower().find("windows")>=0:
        LOG_DIR = "D:/eclipse/cs224n/own/skipmodel/"
    else:
        LOG_DIR = "/home/cc/data/skipmodel/%s/"%dataset
        
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)


    #data存放的是数据集中词汇在one_hoting的index，为下一步embeding_lookup
    #cout存放的是每一个词汇出现的频率
    #dictionary存放的是 词汇：index 字典
    #reverse_dictionary 是 index: 词汇 字典
    #vocabulary_size 总词汇数量
    VOCABLARY_DATA = LOG_DIR + "/vocablary.txt"
    if TYPE == "predict":
        with open(VOCABLARY_DATA) as f:
            cache_data = json.loads(f.read())
            dictionary = cache_data["dictionary"]
            reverse_dictionary = cache_data["reverse_dictionary"]
            vocabulary_size = cache_data["vocabulary_size"]
    else:
        content_lists = []
        if dataset == "sdmz":
            content_lists = get_sdmz()
        elif dataset == "sogou":
            # content_lists = get_sogou_news()
            get_sogou_news()
        # for i in content_lists[0:10]:
        #     print(i)
        data, count, dictionary, reverse_dictionary, vocabulary_size = build_dataset(content_lists)
        with open(VOCABLARY_DATA, "w") as f:
            f.write(json.dumps(dict(
                dictionary = dictionary,
                reverse_dictionary = reverse_dictionary,
                vocabulary_size = vocabulary_size
            )))
    
    #验证数据集， 用于计算 近似词汇 数据信息
    valid_size = 16  # 生成多少个验证数据求其近似词汇
    valid_window = 100  # 验证词汇的index范围
    valid_examples = np.random.choice(valid_window, valid_size, replace=False) #从valid_window大小里面随机选出valid_size个数据出来，array的大小位valid_size

    graph = tf.Graph()
    with graph.as_default():
        with tf.name_scope("inputs"):
            train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
            train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
            valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        with tf.name_scope("embeddings"):
            embeddings = tf.Variable(tf.random_uniform( [vocabulary_size, embedding_size], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        with tf.name_scope("weights"):
            nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))

        with tf.name_scope("biases"):
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        with tf.name_scope("loss"):
            loss = tf.reduce_mean(
                tf.nn.nce_loss(
                    weights=nce_weights,
                    biases=nce_biases,
                    labels=train_labels,
                    inputs=embed,
                    num_sampled=num_sampled,
                    num_classes=vocabulary_size
                )
            )

        tf.summary.scalar('loss', loss)

        with tf.name_scope("optimizer"):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)

        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,
                                            valid_dataset)

        #计算验证数据集的近似词汇
        similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
        merged = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=5)


    with tf.Session(graph=graph) as session:
        writer = tf.summary.FileWriter(LOG_DIR, session.graph)
        init.run()
        print('Initialized')

        #如果存在之前的saver，则还原之前的模型再进行计算
        if TYPE == "predict":
            saver.restore(session, os.path.join(LOG_DIR, 'model.ckpt'))
            while True:
                input_vocab = input("Please input check vocablary: ")
                if input_vocab.strip() == "exit":
                    break
                input_embed = tf.nn.embedding_lookup(normalized_embeddings, [dictionary[input_vocab]])
                input_similarity = tf.matmul(input_embed, normalized_embeddings, transpose_b=True)
                sim = input_similarity.eval()
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[0, :]).argsort()[1:top_k + 1]
                log_str = 'Nearest to %s:' % input_vocab
                for k in xrange(top_k):
                    close_word = reverse_dictionary[str(nearest[k])]
                    log_str = '%s %s,' % (log_str, close_word)
                print(log_str)
        else:
            average_loss = 0
            for step in xrange(num_steps):
                batch_inputs, batch_labels = generate_batch(batch_size, num_skips,
                                                    skip_window)
                feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

                run_metadata = tf.RunMetadata()

                _, summary, loss_val = session.run(
                    [optimizer, merged, loss],
                    feed_dict=feed_dict,
                    run_metadata=run_metadata)
                average_loss += loss_val

                writer.add_summary(summary, step)

                if step == num_steps - 1:
                    writer.add_run_metadata(run_metadata, 'step%d' % step)

                if step % 2000 == 0:
                    if step > 0:
                        average_loss /= 2000
                    # The average loss is an estimate of the loss over the last 2000 batches.
                    print('Average loss at step ', step, ': ', average_loss)
                    average_loss = 0

                if step % 1000 == 0:
                    saver.save(session, os.path.join(LOG_DIR, 'model.ckpt'), global_step=step, write_meta_graph=True, write_state=True)
                if step % 10000 == 0:
                    sim = similarity.eval()
                    for i in xrange(valid_size):
                        valid_word = reverse_dictionary[valid_examples[i]]
                        top_k = 8  # number of nearest neighbors
                        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                        log_str = 'Nearest to %s:' % valid_word
                        for k in xrange(top_k):
                            close_word = reverse_dictionary[nearest[k]]
                            log_str = '%s %s,' % (log_str, close_word)
                        print(log_str)

            final_embeddings = normalized_embeddings.eval()

            saver.save(session, os.path.join(LOG_DIR, 'model.ckpt'))

            # Write corresponding labels for the embeddings.
            with open(LOG_DIR + '/metadata.tsv', 'w', encoding="utf8") as f:
                for i in xrange(vocabulary_size):
                    # print(i)
                    f.write(reverse_dictionary[i] + '\n')

            # Create a configuration for visualizing embeddings with the labels in TensorBoard.
            config = projector.ProjectorConfig()
            embedding_conf = config.embeddings.add()
            embedding_conf.tensor_name = embeddings.name
            embedding_conf.metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')
            projector.visualize_embeddings(writer, config)

            # print("vocabulary size: %d, embedding size: %d"%(vocabulary_size, embeddings.shape[0]))

        writer.close()


    # # pylint: disable=missing-docstring
    # # Function to draw visualization of distance between embeddings.
    # def plot_with_labels(low_dim_embs, labels, filename):
    #     assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    #     plt.figure(figsize=(18, 18))  # in inches
    #     for i, label in enumerate(labels):
    #         x, y = low_dim_embs[i, :]
    #         plt.scatter(x, y)
    #         plt.annotate(
    #             label,
    #             xy=(x, y),
    #             xytext=(5, 2),
    #             textcoords='offset points',
    #             ha='right',
    #             va='bottom')

    #     plt.savefig(filename)
    # print("create TSNE info")
    # try:
    #     # pylint: disable=g-import-not-at-top
    #     from sklearn.manifold import TSNE
    #     import matplotlib.pyplot as plt

    #     tsne = TSNE(
    #         perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    #     plot_only = 500
    #     low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    #     labels = [reverse_dictionary[i] for i in xrange(plot_only)]
    #     plot_with_labels(low_dim_embs, labels, os.path.join(os.gettempdir(), 'tsne.png'))

    # except ImportError as ex:
    #     print('Please install sklearn, matplotlib, and scipy to show embeddings.')
    #     print(ex)
import gensim
import zipfile
import logging
import tensorflow as tf
import codecs

logging.getLogger().setLevel(logging.INFO)

def word2Vec(corpusPath):
  sentences2 = [read_data(corpusPath)]
  logging.critical("Has loaded corpus file and modeling with Word2Vec")
  model = gensim.models.Word2Vec(sentences2, min_count=5, size=300)
  model.save("./data/vec")

# Read the data into a list of strings.
def read_data(filename):
  """Extract the first file enclosed in a zip file as a list of words"""
  #with zipfile.ZipFile(filename) as f:
  #  data = tf.compat.as_str(f.read(f.namelist()[0])).split()
  with codecs.open(filename, "r", encoding="utf-8") as f:
    text = f.read()
    data = tf.compat.as_str(text).split()
  return data

if __name__ == '__main__':
  #word2Vec(corpusPath = None)
  1+1

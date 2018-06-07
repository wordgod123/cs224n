#!python
#coding=utf8

import xml.etree.ElementTree as ET
import os

class xmlParse:
    def __init__(self):
        self.root = None

    def parseXmlFromFile(self, filepath):
        # tree = ET.parse(filepath)
        # self.root = tree.getroot()
        with open(filepath) as f:
            self.parseXmlFromContent(f.read())

    def parseXmlFromContent(self, content):
        self.content = content
        try:
            self.root = ET.fromstring(content.replace("&", "&amp;"))
        except Exception as e:
            print(content)
            raise e
    
    def getContent(self, tag):
        try:
            return self.root.find(tag).text
        except Exception as e:
            print(self.content)
            raise e

def splitXmls(filepath, writeDir="/tmp/splitXmls"):
    if not os.path.exists(writeDir):
        os.makedirs(writeDir)
    content = ''   
    end = False
    x = xmlParse()
    with open(filepath) as fr:
        for line in fr:
            content += line
            if line.strip() == "</doc>":
                # print(content)
                x.parseXmlFromContent(content.replace("&", "&amp;"))
                title = x.getContent("docno")
                with open(writeDir + "/" + title + ".xml", "w") as wf:
                    wf.write(content)
                content = ""

if __name__ == "__main__":
    xmldoc = "/home/cc/data/news_sohusite_xml.full/clean.txt"
    splitXmls(xmldoc, writeDir="/home/cc/data/news_sohusite_xml.full/split")
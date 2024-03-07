# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
import json


class RenthousepriceJsonPipeline(object):
    def __init__(self):
        self.file = open('rent_price.json', 'wb')
        self.file.write(b"[")

    def process_item(self, item, spider):
        content = json.dumps(dict(item), ensure_ascii=False) + "," + "\n"
        self.file.write(content.encode())
        return item

    def close_spider(self, spider):
        self.file.seek(-2, 2)
        self.file.truncate() # 删除多余的一个 ,
        self.file.write(b"]")
        self.file.close()

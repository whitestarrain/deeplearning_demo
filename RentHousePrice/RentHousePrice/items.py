# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class RenthousepriceItem(scrapy.Item):
    title = scrapy.Field()  # 名称
    region = scrapy.Field()  # 位置
    price = scrapy.Field()  # 价格
    house_type = scrapy.Field()  # 户型
    rent_type = scrapy.Field()  # 租住方式
    orientation = scrapy.Field()  # 朝向

    # define the fields for your item here like:
    # name = scrapy.Field()
    pass

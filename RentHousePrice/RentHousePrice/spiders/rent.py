import scrapy
from scrapy.http.cookies import CookieJar


class RentSpider(scrapy.Spider):
    name = 'rent'
    allowed_domains = ['sy.zu.fang.com']
    start_urls = ['http://sy.zu.fang.com/']

    def start_requests(self):
        yield scrapy.Request(url='https://sy.zu.fang.com/house/i3100/?rfss=1-2fa8b7afca4a370575-45',
                             meta={'cookiejar': 1},
                             callback=self.start_parse)
        # 获取cookie

    def start_parse(self, response):
        meta_data = {'cookiejar': response.meta['cookiejar']}
        yield scrapy.Request(url='http://sy.zu.fang.com/', meta=meta_data, callback=self.parse)

    def parse(self, response, **kwargs):
        houses = response.xpath(
            "/html/body/div[@class='wrap']/div[@id='houselistbody']/div/div[@class='houseList']/dl/dd")
        for house in houses:
            title = house.xpath("./p[@class='title']/a/text()").extract()[0]
            region = ""
            for s in house.xpath("./p[3]/a/span/text()").extract():
                region += s
            price = house.xpath("./div[2]/p/span/text()").extract()[0]
            house_type = house.xpath("./p[2]/text()").extract()[1]
            rent_type = house.xpath("./p[2]/text()").extract()[0].replace(" ", "")[2:5]
            orientation = house.xpath("./p[2]/text()").extract()[3][0:2]
            temp = dict(
                title=title,
                region=region,
                price=price,
                house_type=house_type,
                rent_type=rent_type,
                orientation=orientation
            )
            yield temp

        now_page = int(response.xpath("//a[@class='pageNow']/text()").extract()[0])
        if now_page <= 100:
            next_page_uri = "http://sy.zu.fang.com" + \
                            response.xpath("//a[@class='pageNow']/following-sibling::a[1]/@href").extract()[0]
            print("page---" + str(now_page))
            print("next page --" + next_page_uri)
            meta_data = {'cookiejar': response.meta['cookiejar']}
            yield scrapy.Request(next_page_uri, callback=self.parse, meta=meta_data)

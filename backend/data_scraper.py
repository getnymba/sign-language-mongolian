import requests
import os

MONGOLIAN_ALPHABET = ['а', 'б', 'в', 'г', 'д',
                      'е', 'ё', 'ж', 'з', 'и',
                      'й', 'к', 'л', 'м', 'н',
                      'о', 'ө', 'п', 'р', 'с',
                      'т', 'у', 'ү', 'ф', 'х',
                      'ц', 'ч', 'ш', 'щ', 'ъ',
                      'ы', 'ь', 'э', 'ю', 'я']

url = 'https://mnsl.mn/wp-content/uploads/2024/07/{index}.-{value}-300x225.jpg'
for index, value in enumerate(MONGOLIAN_ALPHABET):
    pic = url.format(index=index+1, value=value.upper())
    response = requests.get(pic, stream=True).content
    mypath = './dataset/' + value.upper()
    if not os.path.isdir(mypath):
        os.makedirs(mypath)
    with open(mypath + '/' + '300x225.jpg', 'wb') as out_file:
        out_file.write(response)

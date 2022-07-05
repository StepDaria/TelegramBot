import styletransfer  # Перенос стиля
import GANStyleTransfer  # Превращение лошади(зебры) в зебру(лошадь) на CycleGAN


import torchvision.transforms as tt
from aiogram import Bot, Dispatcher, executor, types
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
import io
from io import BytesIO
from PIL import Image

TOKEN = 'your_token_here'
BOT = Bot(token=TOKEN)
STORAGE = MemoryStorage()
DP = Dispatcher(BOT, storage=STORAGE)


class SendingPhoto(StatesGroup):
    """
    Перечень состояний для корректной работы бота и сохранении необходимой информации
    """
    sending_content_photo = State()
    sending_style_photo = State()
    sending_horse_photo = State()
    sending_zebra_photo = State()
    more = State()


def to_PIL(img):
    """
    Функция, переводящая torch.Tensor в байты
    """
    converter = tt.ToPILImage()
    image = img.cpu()
    image = image.squeeze(0)
    image = converter(image)
    return image


def to_bytes(img):
    """
    Функция, переводящая байты в PIL изображение
    """
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr


def keyboard():
    """
    Фунция, создающая виртуальную клавиатуру для ответа
    """
    markup = types.InlineKeyboardMarkup()
    markup.add(types.InlineKeyboardButton(text='Перенести стиль', callback_data='style'))
    markup.add(types.InlineKeyboardButton(text='Лошадь -> зебра', callback_data='h2z'))
    markup.add(types.InlineKeyboardButton(text='Зебра -> лошадь', callback_data='z2h'))
    return markup


@DP.message_handler(commands=['start'], state='*')
async def start(message):
    await BOT.send_message(message.chat.id, 'Привет. Меня зовут StyleTransferBot,\
     я могу преносить стиль с одной фотографии на другую. Вот как это выглядит:')
    photo = open(r'images/style_transfer.jpg', 'rb')  # пример работы функции "Перенести стиль"
    await BOT.send_photo(message.chat.id, photo)

    await BOT.send_message(message.chat.id, 'А еще я умею превращать лошадей в зебры и обратно:')
    photo = open(r'images/horse_to_zebra.png', 'rb')
    await BOT.send_photo(message.chat.id, photo) # пример работы функции "Лошадь-зебра"
    await BOT.send_message(message.chat.id, 'Хотите попробовать?', reply_markup=keyboard())
    await SendingPhoto.more.set()  # устанавливаем состояние ожидания ответа


@DP.message_handler(commands=['help'], state='*')
async def help(message):
    await BOT.send_message(message.chat.id, '/start - запуск бота\n/select - выбор опции\n/help - помощь')
    await SendingPhoto.more.set()


@DP.message_handler(commands=['select'], state='*')
async def select(message):
    await BOT.send_message(message.chat.id, 'Выберите действие', reply_markup=keyboard())
    await SendingPhoto.more.set()


@DP.callback_query_handler(lambda call: True, state='*')
async def answer(call):
    if call.data == 'style':
        await BOT.send_message(call.message.chat.id, 'Отправьте картинку, которую нужно изменить')
        await SendingPhoto.sending_content_photo.set()  # устанавливаем состояние "Перенести стиль" и ждем фото

    elif call.data == 'h2z':
        await BOT.send_message(call.message.chat.id, 'Отправьте картинку лошади')
        await SendingPhoto.sending_horse_photo.set()  # устанавливаем состояние "Лошадь-зебра" и ждем фото

    elif call.data == 'z2h':
        await BOT.send_message(call.message.chat.id, 'Отправьте картинку зебры')
        await SendingPhoto.sending_zebra_photo.set()  # устанавливаем состояние "Зебра-лошадь" и ждем фото


@DP.message_handler(content_types=['photo'], state=SendingPhoto.sending_content_photo)
async def content_photo(message: types.Message, state: FSMContext):
    """
    Получение фото-контента для переноса стиля
    """
    file = await BOT.get_file(message.photo[-1].file_id)
    downloaded_file = await BOT.download_file(file.file_path)
    async with state.proxy() as data:
        data['content'] = downloaded_file.getvalue()  # записываем полученное фото в текущее состоние,
                                                      # чтоб работать с ним дальше

    await BOT.send_message(message.chat.id, 'Отправьте картинку стиля')
    await SendingPhoto.sending_style_photo.set()  # Устанавливаем состояние ожидания фото-стиля


@DP.message_handler(content_types=['photo'], state=SendingPhoto.sending_style_photo)
async def style_photo(message: types.Message, state: FSMContext):
    """
    Получение фото-стиля для переноса стиля и сам перенос стиля
    """
    file = await BOT.get_file(message.photo[-1].file_id)
    downloaded_file = await BOT.download_file(file.file_path)
    async with state.proxy() as data:
        data['style'] = downloaded_file.getvalue() # записываем полученное фото в текущее состоние

    # Превращаем байты в PIL-изображение
    content_img = Image.open(io.BytesIO(data['content'])).convert('RGB')
    style_img = Image.open(io.BytesIO(data['style'])).convert('RGB')
    # Превращаем PIL в тензоры
    content_img, content_size = styletransfer.image_loader(content_img, return_size=True)
    style_img = styletransfer.image_loader(style_img, image_size=content_size)

    input_img = content_img.clone()
    output_img = styletransfer.run_style_transfer(content_img, style_img, input_img)  # Переносим стиль
    output_img = to_PIL(output_img)  # Превращаем тензор в PIL-изображение
    output_img = to_bytes(output_img) # Превращаем PIL-изображение в байты
    await BOT.send_photo(message.chat.id, output_img, reply_markup=keyboard())  # Отправляем изображение
    await state.finish()  # Чистим состояние
    await SendingPhoto.more.set()  # Устанавливаем состояние ожидания выбора следующего действия


@DP.message_handler(content_types=['photo'], state=SendingPhoto.sending_horse_photo)
async def h2z(message: types.Message):
    """
    Получение фото лошади и превращение в зебру
    """
    file = await BOT.get_file(message.photo[-1].file_id)
    downloaded_file = await BOT.download_file(file.file_path)
    downloaded_file = downloaded_file.getvalue()  # Получили байтовое изображение
    pil_img = Image.open(io.BytesIO(downloaded_file)).convert('RGB')  # Трансформировали в PIL
    fake_image = GANStyleTransfer.run_transform(pil_img, 'h2z')  # Превращение в зебру
    photo = to_bytes(fake_image)  # Превращаем PIL-изображение в байты
    await BOT.send_photo(message.chat.id, photo, reply_markup=keyboard())  # Отправляем изображение
    await SendingPhoto.more.set()  # Устанавливаем состояние ожидания выбора следующего действия


@DP.message_handler(content_types=['photo'], state=SendingPhoto.sending_zebra_photo)
async def z2h(message: types.Message):
    file = await BOT.get_file(message.photo[-1].file_id)
    downloaded_file = await BOT.download_file(file.file_path)
    downloaded_file = downloaded_file.getvalue()  # Получили байтовое изображение
    pil_img = Image.open(io.BytesIO(downloaded_file)).convert('RGB')  # Трансформировали в PIL
    fake_image = GANStyleTransfer.run_transform(pil_img, 'z2h')  # Превращение в лошадь
    photo = to_bytes(fake_image)  # Превращаем PIL-изображение в байты
    await BOT.send_photo(message.chat.id, photo, reply_markup=keyboard())  # Отправляем изображение
    await SendingPhoto.more.set()  # Устанавливаем состояние ожидания выбора следующего действия

executor.start_polling(DP, skip_updates=True)

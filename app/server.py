import aiohttp
import asyncio
import uvicorn

# from fastai import *
# from fastai.vision import *
from pathlib import Path
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

from utils import *

mask_detector_file_url = 'https://www.dropbox.com/s/xl90ovl3bwgg1ql/mask_detector-model.pkl?dl=1'
mask_detector_file_name = 'mask_detector-model.pkl'


masked_gender_detector_file_url  = "https://www.dropbox.com/s/9velh2c5ql194mx/gender_mask-model.pkl?dl=1" 
masked_gender_detector_file_name = "gender_mask-model.pkl"


non_masked_gender_detector_file_url  = "https://www.dropbox.com/s/dhb7qgnrur26dsi/gender_no_mask-model.pkl?dl=1"
non_masked_gender_detector_file_name = "gender_no_mask-model.pkl"



# classes = ['mask', 'no_mask']
path = Path(__file__).parent

app = Starlette(debug=True)
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(mask_detector_file_url, path / mask_detector_file_name)
    try:
        learn = load_learner(path/mask_detector_file_name)
        return learn
        # return learn, learn_mask_gender, learn_nomask_gender
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


async def setup_learner_maskgender():
    await download_file(masked_gender_detector_file_url, path / masked_gender_detector_file_name)
    try:
        learn_mask_gender   = load_learner(path/masked_gender_detector_file_name)
        return learn_mask_gender
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


async def setup_learner_nomaskgender():
    await download_file(non_masked_gender_detector_file_url, path / non_masked_gender_detector_file_name)
    try:
        learn_nomask_gender   = load_learner(path/non_masked_gender_detector_file_name)
        return learn_nomask_gender
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


res_prototxt = "deploy.prototxt.txt"
res_model    = "res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(f"app/{res_prototxt}", f"app/{res_model}")
loop = asyncio.get_event_loop()

tasks = [asyncio.ensure_future(setup_learner())]
mask_detector = loop.run_until_complete(asyncio.gather(*tasks))[0]

tasks_2 = [asyncio.ensure_future(setup_learner_maskgender())]
mask_gender = loop.run_until_complete(asyncio.gather(*tasks_2))[0]

tasks_3 = [asyncio.ensure_future(setup_learner_nomaskgender())]
mask_nogender = loop.run_until_complete(asyncio.gather(*tasks_3))[0]



loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    # print(f"The img_data is {img_data} and type is {type(img_data)}")
    img_bytes = await (img_data['file'].read()) #bytes data
    ### checking face detection here
    pil_image = Image.open(BytesIO(img_bytes)) #get PIL Image
    opencvImage = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR) #change to CV compatible numpyarray
    # img = np.array(Image.open(BytesIO(img_bytes))) #NP array of PIL image
    processed_file = process_image(opencvImage, net)
    # print(f"The processed_file is {processed_file} and type is {type(processed_file)}")
    # prediction = learn.predict(img)[0]
    is_masked = mask_detector.predict(processed_file)[0]
    if is_masked == 'True':
        prediction = mask_gender.predict(processed_file)[0]
    elif is_masked == 'False':
        prediction = mask_nogender.predict(processed_file)[0]
    print(f"The prediction is {prediction}")
    return JSONResponse({'result': str(prediction)})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")

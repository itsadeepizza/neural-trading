import dotenv
import os
import asyncio
from websockets import connect
from datetime import datetime
import json

dotenv.load_dotenv()
API_KEY = os.getenv('API_KEY')

# wss://ws.coincap.io/prices?assets=bitcoin,ethereum,monero,litecoin

uri = f"wss://ws.coincap.io/prices?assets=dogecoin,bitcoin,ethereum,monero,litecoin&api_key={API_KEY}"
# uri = f"wss://ws.coincap.io/prices?assets=ALL&api_key={API_KEY}"
async def get_prices(q):
    async with connect(uri) as websocket:
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S:%f")
            # add to queue
            await q.put((data, timestamp))
            # print(f"{timestamp} - {message}")




async def store_prices(q, csv_name):
    with open(csv_name, 'a') as f:
        batch_size = 10
        while True:
            batch_size += 1
            data, timestamp = await q.get()
            print(f"{timestamp} - {data}")
            for coin, price in data.items():
                f.write(f"{timestamp};{coin};{price}\n")
                pass
            # Close the file and reopen after 10 writes
            if batch_size % 100 == 0:
                f.close()
                f = open(csv_name, 'a')
            q.task_done()

async def main(csv_name):
    # Create a new csv file to store the data
    q = asyncio.Queue()
    task_producer = asyncio.create_task(get_prices(q))
    task_consumer = asyncio.create_task(store_prices(q, csv_name))
    await task_producer
    await q.join()


csv_name = f"dataset/prices_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
with open(csv_name, 'a') as f:
    f.write("timestamp;coin;price\n")
asyncio.run(main(csv_name))


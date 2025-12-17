import uvicorn

if __name__ == "__main__":
    uvicorn.run("amazon_sales_ml.api.app:app", host="0.0.0.0", port=8000, reload=True)

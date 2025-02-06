#!/bin/bash

uvicorn BackgroundRemover:app --host 0.0.0.0 --port $PORT

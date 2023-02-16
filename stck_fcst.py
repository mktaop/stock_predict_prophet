#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 20:25:58 2023

@author: avi_patel
"""


import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import date, timedelta
from prophet import Prophet


#....page title, icon and layout set up and display
page_title="Predict Stock Price - You provide the stock symbol, using Prophet we will predict!"
page_icon=":chart_with_upwards_trend:"
layout="centered"
st.set_page_config(page_title=page_title, page_icon=page_icon, layout=layout)
st.title(page_icon + " " + page_title + " " + page_icon)
#....user input:  get a stock symbol....
stock_symbol = st.text_input('Stock symbol, please:  ')

#...once a symbol is entered and user presses return, create forecast and graph
if stock_symbol != '':
    
    today = date.today()
    start_period = today - timedelta(days=1)
    lookback=150
    lookahead=6
    end_period= today - timedelta(days=lookback)
    
    df = pd.DataFrame(yf.download(stock_symbol,start=end_period,end=start_period,progress=False))
    df=df.reset_index()
    data = df[["Date","Close"]]
    data = data.rename(columns = {"Date":"ds","Close":"y"}) 
    m = Prophet(growth='linear',
                changepoints=None,
                n_changepoints=25,
                changepoint_range=0.8,
                yearly_seasonality='auto',
                weekly_seasonality='auto',
                daily_seasonality = True,
                holidays=None,
                seasonality_mode='additive',
                seasonality_prior_scale=10.0,) # the Prophet class (model)
    m.fit(data) # fit the model using all data
    
    forecast_pd = m.make_future_dataframe(periods=lookahead, freq='MS')
    forecast = m.predict(forecast_pd)
    plt.figure(figsize=(18, 6))
    fig=m.plot(forecast, xlabel = 'Date', ylabel = 'Close')
    plt_title='Closing price forecast for: ' + '"' + stock_symbol + '"'
    plt.title(plt_title)
    st.write(fig)
    
else:
    pass


import datetime as dt
import yfinance as yf

############## Search company name from 'finance.yahoo.com' and change the date range accordingly #############
company = "TCS.NS"
start = dt.datetime(2018, 10, 6)
end = dt.datetime(2024, 10, 8)

##################### Download the data ####################
data = yf.download(company, start=start, end=end)

################ Display the data ####################
print(data)

############### Convert the index to just date (removing the time) ###################
data.index = data.index.date

###################### Save the data to an Excel file ###################
file_name = "TCS_data.xlsx" # we can change excel name
data.to_excel(file_name, engine='openpyxl')

print(f"Data saved to {file_name}") # status ######

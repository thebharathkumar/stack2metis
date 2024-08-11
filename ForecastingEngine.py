import pandas as pd

from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import warnings
from datetime import datetime
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import pickle
from xgboost import plot_importance, plot_tree
import re
import yfinance as yf
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import os
class FinancialModeling:
    def __init__(self,analysis_type, company_name, **kwargs):
        self.company_name = ""
        self.company_code = ""

        self.sheets_data = None
        # self.get_sheets_data()
        # self.get_company(company_name)
        self.dates = None
        
        self.historicals = None
        
        self.historical_items = None
        self.trend_pipelines = None
        self.xgboost_models = None
        # self.trends = None
        self.xgboost_predictions = None
        
        self.final_predictions = None
        self.ufcf = None
        
        self.cmd = ""
        
        self.Equity = None
        self.Debt = None
        self.InterestExpense = None
        self.UtreBond = 0.044
        self.beta = None
        self.marketriskpremium = 0.0397
        self.TGR = 0.03
        self.Shares = None
        self.Cash = None
        self.Taxes = 0.21
        self.Tax = 0.21
        self.training_data = None

        self.monte_carlo_simulations = None
        self.fcf = None
        self.dcf = None
        self.Wacc = None
        self.wacc = None
        self.data = None
        self.prepped_data, self.trends = None, None
        self.training_data = None
        # self.monte_carlo_simulations = None
        self.cca_company_name_A = ""
        self.cca_company_name_B = ""
        self.cca_company_code_A = ""
        self.cca_company_code_B = ""
        self.cca_company_names_codes = None
        self.cca_data = None
        self.comp_avg = None
        self.evaluation = None
        self.cca_company_historicals_A = None
        self.cca_company_historicals_B = None
        self.valuation = None
        self.cca__historicals_data = None
        self.cca_companies = None
        self.cca_data = None
        self.run_analysis(analysis_type, company_name, **kwargs)

    def run_analysis(self, analysis_type, companies, **kwargs):
        try:
            if analysis_type == "DCF":
                if len(companies) == 1:
                    self.get_sheets_data(companies[0])
                    self.get_company(companies[0])
                    self.get_historicals(companies[0])
                    self.prepped_data, self.trends = self.prep_data(self.historicals)
                    self.training_data = self.prepped_data
                    self.return_dcf_and_fcf_montecarlo()
            if analysis_type == "FCF":
                if len(companies) == 1:
                    self.get_sheets_data(companies[0])
                    self.get_company(companies[0])
                    self.get_historicals(companies[0])
                    self.prepped_data, self.trends = self.prep_data(self.historicals)
                    self.training_data = self.prepped_data
                    self.return_fcf_montecarlo()
            if analysis_type == "WACC":
                if len(companies) == 1:
                    self.get_sheets_data(companies[0])
                    self.get_company(companies[0])
                    self.get_historicals(companies[0])
                    self.prepped_data, self.trends = self.prep_data(self.historicals)
                    self.training_data = self.prepped_data
                    self.return_wacc()
            if analysis_type == "CCA":
                if len(companies) > 1:
                    # self.get_company(companies[0])
                    # self.get_historicals(companies)
                    self.cca_company_names_codes=self.cca_get_companies(companies)
                    self.cca_data=self.cca_get_historicals(companies)
                    self.return_cca()
                    # self.training_data = self.prepped_data
                    # self.calculate_wacc()
        except:
            raise ValueError(f"Data for '{self.company_name}' is not available.")

    def return_dcf_and_fcf_montecarlo(self):
        try:
            if self.dcf is None:
                self.calculate_dcf()
            self.monte_carlo_simulation()
            return self.dcf, self.fcf, self.monte_carlo_simulations
        except:
            raise ValueError(f"Data for '{self.company_name}' is not available.")
    def return_fcf_montecarlo(self):
        try:
            if self.fcf is None:
                self.calculate_fcf()
            self.monte_carlo_simulation()
            return self.fcf, self.monte_carlo_simulations
        except:
            raise ValueError(f"Data for '{self.company_name}' is not available.")
    def return_wacc(self):
        try:
            if self.Wacc is None:
                self.calculate_wacc()
            return self.Wacc
        except:
            raise ValueError(f"Data for '{self.company_name}' is not available.")
    def return_historicals(self):
        try:
            if self.historicals is None:
                self.get_historicals(self.company_name)
            return self.historicals
        except:
            raise ValueError(f"Data for '{self.company_name}' is not available.")
        
    def return_cca(self):
        try:
            if self.evaluation is None:
                self.cca()
            return self.cca_data, self.comp_avg, self.evaluation
        except:
            raise ValueError(f"Data for '{self.company_name}' is not available.")

    
    def get_sheets_data(self,companyname):
        try:
            files_path = r"/Users/joshna/Desktop/Metis/Financials"
            filepath = os.path.join(files_path, f"{companyname} Financials.xlsx")
            file_path = r""+filepath
            excel_workbook = pd.ExcelFile(file_path)
            # Dictionary to hold data from each sheet
            sheets_data = {}
            for sheet_name in excel_workbook.sheet_names:
                sheet_data = pd.read_excel(file_path, sheet_name=sheet_name)
                sheets_data[sheet_name] = sheet_data
            self.sheets_data = sheets_data
            print("Sheets data acquired")
        except:
            raise ValueError(f"Data for '{self.company_name}' is not available.")
    def get_company(self, company_name1):
        def extract_data_between_parentheses(text):
            # Define the regular expression pattern to match text between parentheses
            sheets_data = self.sheets_data
            pattern = r'\((.*?)\)'
            
            # Use re.findall to find all matches of the pattern in the text
            matches = re.findall(pattern, text)
            
            return matches
        company_name = ""
        company_code = ""
        sheets_data = self.sheets_data
        for sheet_name, data in sheets_data.items():
            if sheet_name == 'Income Statement':
                company_name = data[data.iloc[:, 0] == "Company Name"].iloc[:, 1:2].values
        print(company_name[0][0])
        company_name = company_name[0][0]
        company_code = extract_data_between_parentheses(company_name)[0]
        self.company_name = company_name
        self.company_code = company_code
    def get_historicals(self,company_name):
        try:
            sheets_data = self.sheets_data
            items = []
            
            dates = 0
            # dates = dates[0]
            for sheet_name, data in sheets_data.items():
                if sheet_name == 'Income Statement':
                    dates = data[data.iloc[:, 0] == "Field Name"].iloc[:, 1:].values.tolist()
                    dates = dates[0]
                    self.dates = dates
                    revenue_data = data[data.iloc[:, 0] == "Revenue from Business Activities - Total"].iloc[:, 1:].values
                    revenue = revenue_data[~np.isnan(revenue_data.astype(float))].tolist()
                    if revenue:
                        items.append(("Revenue", revenue))
                    
                    operating_expense_data = data[data.iloc[:, 0] == "Operating Expenses - Total"].iloc[:, 1:].values
                    operating_expense = operating_expense_data[~np.isnan(operating_expense_data.astype(float))].tolist()
                    if operating_expense:
                        items.append(("Operating_Expense", operating_expense))
                    
                    interest_expense_data = data[data.iloc[:, 0] == "Interest Expense - Net of (Interest Income)"].iloc[:, 1:].values
                    interest_expense = interest_expense_data[~np.isnan(interest_expense_data.astype(float))].tolist()
                    interestexpense1 = data[data.iloc[:, 0] == "Interest Expense"].iloc[:, 1:5].values
                    self.interestexpense= interestexpense1[0]
                    if interest_expense:
                        items.append(("Interest_Expense", interest_expense))
                    
                    ebit_data = data[data.iloc[:, 0] == "Earnings before Interest & Taxes (EBIT)"].iloc[:, 1:].values
                    ebit = ebit_data[~np.isnan(ebit_data.astype(float))].tolist()
                    if ebit:
                        items.append(("EBIT", ebit))
                    
                    ebitda_data = data[data.iloc[:, 0] == "Earnings before Interest, Taxes, Depreciation & Amortization (EBITDA)"].iloc[:, 1:].values
                    ebitda = ebitda_data[~np.isnan(ebitda_data.astype(float))].tolist()
                    if ebitda:
                        items.append(("EBITDA", ebitda))
                    
                    SGandA_data = data[data.iloc[:, 0] == "Selling, General & Administrative Expenses - Total"].iloc[:, 1:].values
                    SGandA = SGandA_data[~np.isnan(SGandA_data.astype(float))].tolist()
                    if SGandA:
                        items.append(("SGandA", SGandA))
                    Shares = data[data.iloc[:, 0] == "Common Shares - Issued - Total"].iloc[:, 1:2].values
                
                elif sheet_name == 'Cash Flow':
                    capital_expenditures_data = data[data.iloc[:, 0] == "Capital Expenditures - Total"].iloc[:, 1:].values
                    capital_expenditures = capital_expenditures_data[~np.isnan(capital_expenditures_data.astype(float))].tolist()
                    if capital_expenditures:
                        items.append(("Capital_Expenditures", capital_expenditures))
                
                elif sheet_name == 'Financial Summary':
                    DandA_data = data[data.iloc[:, 0] == "Depreciation, Depletion & Amortization including Impairment - Cash Flow - to Reconcile"].iloc[:, 1:].values
                    DandA = DandA_data[~np.isnan(DandA_data.astype(float))].tolist()
                    if DandA:
                        items.append(("DandA", DandA))
                    
                    Gprofit_data = data[data.iloc[:, 0] == "Gross Profit - Industrials/Property - Total"].iloc[:, 1:].values
                    Gprofit = Gprofit_data[~np.isnan(Gprofit_data.astype(float))].tolist()
                    if Gprofit:
                        items.append(("Gross_Profit", Gprofit))
                    Shares = data[data.iloc[:, 0] == "Common Shares - Outstanding - Total"].iloc[:, 1:2].values
                    self.Shares = Shares[0]
                elif sheet_name == 'Balance Sheet':
                    totalliabilities_data = data[data.iloc[:, 0] == "Total Current Liabilities"].iloc[:, 1:].values
                    totalliabilities = totalliabilities_data[~np.isnan(totalliabilities_data.astype(float))].tolist()
                    Equity = data[data.iloc[:, 0] == "Total Shareholders' Equity - including Minority Interest & Hybrid Debt"].iloc[:, 1:5].values
                    Debt = data[data.iloc[:, 0] == "Debt - Total"].iloc[:, 1:5].values
                    Cash = data[data.iloc[:, 0] == "Cash & Cash Equivalents"].iloc[:, 1:2].values
                    self. Equity = Equity[0]
                    self.Debt = Debt[0]
                    self.Cash = Cash[0]
                    if totalliabilities:
                        items.append(("Total_Current_Liabilities", totalliabilities))
                    
                    totalassets_data = data[data.iloc[:, 0] == "Total Current Assets"].iloc[:, 1:].values
                    totalassets = totalassets_data[~np.isnan(totalassets_data.astype(float))].tolist()
                    if totalassets:
                        items.append(("Total_Current_Assets", totalassets))
            l = min(len(v) for i,v in items)
            #print(len(v) for i, v in items)
            items1 = items
            items = []
            for i,v in (items1):
                items.append((i, v[:l]))
                #print(len(items[-1][1]))
            #print(items)
            dates = dates[:l]
            self.dates = dates
            item_dict = dict(items)
            # print(item_dict)
            tca = item_dict.get('Total_Current_Assets')
            tcl = item_dict.get('Total_Current_Liabilities')
            dna = item_dict.get('DandA')
            currentratio = [i/j for i,j in zip(tca,tcl)]
            # print("tca= ",tca, "\ntcl=", tcl,"\ncurrentrati=",currentratio)
            gp = item_dict.get('Gross_Profit')
            r = item_dict.get('Revenue')
            grossmargin = [i/j for i,j in zip(gp,r)]
            assetturnover = [i/j for i,j in zip(r,tca)]
            dbr = [i/j for i, j in zip(dna,r)]
            #print(dbr)
            items.append(("Current_Ratio", currentratio))
            items.append(("Gross_Margin", grossmargin))
            
            items.append(("Asset_Turnover", assetturnover))
            items.append(("DbR", dbr))
            self. Equity = Equity[0]
            self.Debt = Debt[0]
            print(Debt[0], self.Debt)
            
            self.Cash = Cash[0]
            self.historicals = dict(items)
            self.historical_items = items
        except:
            raise ValueError(f"Data for '{self.company_name}' is not available.")
        
    # def fetch_data(self, company_name):
        
    #     except:
    #         raise ValueError(f"Data for '{company_name}' is not available.")
    def prep_data(self,data1):
        # Simulated data fetching based on company_name
        try:
            items = self.historical_items
            dates = self.dates
            data_frames = []
            for itemname, itemvalues in items:
                if itemname:
                    data = pd.DataFrame({
                        "date": dates,
                        "target": itemvalues,
                        "Value": itemname
                    })
                    #print(itemname)
                    data['date'] = pd.to_datetime(data['date'], dayfirst=True)
                    data['target'] = data['target'].interpolate(method='linear', limit_direction='both')
                    df_reversed = data.iloc[::-1].reset_index(drop=True)
                    data_frames.append(df_reversed)
                if itemname in ['Gross_Margin', 'Asset_Turnover', 'Current_Ratio']:
                    data = pd.DataFrame({
                        itemname: itemvalues,
                    })
                    data[itemname] = data[itemname].interpolate(method='linear', limit_direction='both')
                    df_reversed = data.iloc[::-1].reset_index(drop=True)
                    
                    if itemname == 'Gross_Margin':
                        data_frames[2] = pd.concat([data_frames[2], df_reversed], axis=1)
                    elif itemname == 'Asset_Turnover':
                        data_frames[2] = pd.concat([data_frames[2], df_reversed], axis=1)
                    elif itemname == 'Current_Ratio':
                        data_frames[8] = pd.concat([data_frames[8], df_reversed], axis=1)
                        data_frames[9] = pd.concat([data_frames[9], df_reversed], axis=1)
            for i, df in enumerate(data_frames):
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                
                # Infer and set the frequency of the index
                df.index.freq = pd.infer_freq(df.index)
            data_frames, trend_p = self.get_trends(data_frames)
            def create_lag_features(data, target_column, lags):
                df = data.copy()
                #print(df.head(2))
                for lag in range(1, lags + 1):
                    
                    # df[f'target_lag_{i}'] = df['target'].shift(i)
                    if target_column == 'Asset_Turnover':
                        df[f'{'Asset_Turnover'}_lag_{lag}'] = df[target_column].shift(lag)
                    elif target_column == 'Current_Ratio':
                        df[f'{'Current_Ratio'}_lag_{lag}'] = df[target_column].shift(lag)
                    else:
                
                        df[f'trend_lag_{lag}'] = df['trend'].shift(lag)
                        df[f'seasonal_lag_{lag}'] = df['seasonal'].shift(lag)
                        df[f'resid_lag_{lag}'] = df['resid'].shift(lag)        
                        df[f'{df['Value'][0] if target_column !='Gross_Margin' else 'Gross_Margin'}_lag_{lag}'] = df[target_column].shift(lag)
                return df
            
            # Assuming 'Revenue' is the target variable we want to predict
            data_with_lag = []
            for dataframe in data_frames:
                # print(dataframe['Value'][0])
                data_with_lags = create_lag_features(dataframe, 'target', lags=1)
                data_with_lag.append(data_with_lags)
            data_with_lags = create_lag_features(data_with_lag[2], 'Gross_Margin', lags=1)
            data_with_lag.append(data_with_lags)
            data_with_lags = create_lag_features(data_with_lag[15], 'Asset_Turnover', lags=1)
            data_with_lag.append(data_with_lags)
            data_with_lags = create_lag_features(data_with_lag[15], 'Asset_Turnover', lags=1)
            data_with_lag.append(data_with_lags)
            data_with_lags = create_lag_features(data_with_lag[8], 'Current_Ratio', lags=1)
            data_with_lag.append(data_with_lags)
            data_with_lags = create_lag_features(data_with_lag[9], 'Current_Ratio', lags=1)
            data_with_lag.append(data_with_lags)
            datatrain = []
            datatarget = []
            df1 = data_with_lag.copy()
            datatrain = [df.drop(columns = ['target']) for df in df1]
            
            datatrain = [df.drop(columns = ['Value']) for df in datatrain]
            
            # datatrain = [df.drop(columns = ['date']) for df in datatrain]
            #print(datatrain[13].head())
            datatrain[15].drop(columns = ['Gross_Margin'], inplace=True)
            datatrain[15].drop(columns=['Asset_Turnover'], inplace=True)
            datatrain[16].drop(columns = ['Gross_Margin'], inplace=True)
            datatrain[16].drop(columns=['Asset_Turnover'], inplace=True)
            # datatrain[14].drop(columns=['Gross_Margin_lag_1'], inplace=True)
            datatrain[17].drop(columns=['Gross_Margin_lag_1'], inplace=True)
            datatrain[17].drop(columns=['Gross_Margin'], inplace=True)
            datatrain[17].drop(columns=['Asset_Turnover'], inplace=True)
            datatrain[18].drop(columns=['Current_Ratio'], inplace=True)
            datatrain[19].drop(columns=['Current_Ratio'], inplace=True)
            datatrain[2].drop(columns=['Gross_Margin'], inplace=True)
            datatrain[2].drop(columns=['Asset_Turnover'], inplace=True)
            if 'Asset_Turnover_lag_1' in datatrain[2].columns:
                datatrain[2].drop(columns=['Asset_Turnover_lag_1'], inplace=True)
            for i in range(len(datatrain)):
                datatrain[i].drop(columns=['trend'], inplace=True)
                datatrain[i].drop(columns=['seasonal'], inplace=True)
                # datatrain[i].drop(columns=['trend_lag_1'], inplace=True)    
                datatrain[i].drop(columns=['seasonal_lag_1'], inplace=True)
                datatrain[i].drop(columns=['resid_lag_1'], inplace=True)
                datatrain[i].drop(columns=['resid'], inplace=True)
            
            #print(datatrain[14].head())
            datatarget = [df['target'] for df in df1]
            xtrain, ytrain, xtest, ytest = [],[],[],[] 
            xtrain1, ytrain1, xtest1, ytest1 = [],[],[],[] 
            xtrain2, ytrain2, xtest2, ytest2 = [],[],[],[]
            xtrain3, ytrain3, xtest3, ytest3 = [],[],[],[]
            test_train = {}
            # print(trend_p)
            for i, (x,y) in enumerate(zip(datatrain, datatarget)):
                # if x['date'][0]>'2015-01-01':
                # print(datatrain[i].head(2), datatarget[i].head(2), x.head(2), y.head(2))
                x1 = x[(x.index>pd.Timestamp('2019-01-01'))]
                start_index = x.index.get_loc(x1.index[0])
            
                y1 = y[start_index:start_index+len(x1)]
                #print('\n',x1, y1)
                x1 = x1[x1.index<pd.Timestamp('2023-01-01')]
                
                y1 = y1[:len(x1)]
                #print('\n',x1, y1)
                
                xtrain.append(x1)
                ytrain.append(y1)
                x1 = x[(x.index>pd.Timestamp('2015-01-01'))]
                start_index = x.index.get_loc(x1.index[0])
            
                y1 = y[start_index:start_index+len(x1)]
                # y1 = y[:len(x1)]
                #print('a \n',x1, y1)
                
                x1 = x1[x1.index<pd.Timestamp('2023-01-01')]
                
                y1 = y1[:len(x1)]
                #print('a \n',x1, y1)
                
                xtrain1.append(x1)
                ytrain1.append(y1)
                x1 = x[x.index>pd.Timestamp('2023-01-01')]
                start_index = x.index.get_loc(x1.index[0])
            
                y1 = y[start_index:start_index+len(x1)]
                
                # y1 = y[:len(x1)]
                #print('b \n',x1, y1)
                x1 = x1[x1.index<pd.Timestamp('2024-01-01')]
                
                y1 = y1[:len(x1)]
                #print('b \n',x1, y1)
                xtest.append(x1)
                ytest.append(y1)
                # x1 = x[x.index>pd.Timestamp('2023-01-01')
                # y1 = y[:len(x1)]
                # x1 = x1[x1.index<pd.Timestamp('2024-01-01')]
                
                # y1 = y1[:len(x1)]
                xtest1.append(x[x.index>pd.Timestamp('2023-01-01')])
                ytest1.append(y[:len(x[x.index>pd.Timestamp('2023-01-01')])])
                # xtest2.append(x[x.index>'2023-01-01')
                # ytest2.append(y[:len(x[x.index>'2023-01-01']])
                xtrain3.append(x[x.index>pd.Timestamp('2015-01-01')])
                ytrain3.append(y[:len(x[x.index>pd.Timestamp('2015-01-01')])])
                    
       
                # else:
                # xtrain.append(x.head(len(x)-24))
                # ytrain.append(y.head(len(x)-24))
                # xtrain1.append(x.head(len(x)-24))
                # ytrain1.append(y.head(len(x)-24))
                # xtrain2.append(x.head(len(x)-24))
                # ytrain2.append(y.head(len(x)-24))
                # xtrain3.append(x.head(len(x)-20))
                # ytrain3.append(y.head(len(x)-20))
                xtest3.append(x.tail(20))
                ytest3.append(y.tail(20))
                # print(type(xtrain3), type(ytrain3),  type(x[x.index>pd.Timestamp('2023-01-01')]))
                test_train[data_with_lag[i]['Value'].iloc[0]] = (xtrain3[-1], ytrain3[-1], xtest3[-1], ytest3[-1], trend_p[data_with_lag[i]['Value'].iloc[0] if data_with_lag[i]['Value'].iloc[0] in trend_p.keys() else [0]])
                print("datapreppe")
            self.trends = trend_p
            print(self.trends, trend_p)
            return test_train, trend_p  

        except:
            raise ValueError(f"Data for '{self.company_name}' is not available.")


    def get_trends(self, data_frames):
        try:
            
            trend1 = []
            for i, df in enumerate(data_frames):
                # df['date'] = pd.to_datetime(df['date'])
                
                # df.set_index('date', inplace=True)
                
                # Infer and set the frequency of the index
                df.index.freq = pd.infer_freq(df.index)
                decomposition = sm.tsa.seasonal_decompose(df['target'], model='additive', period = 4 if len(df)<16 else 8)
                trend = decomposition.trend
                seasonal = decomposition.seasonal
                residual = decomposition.resid
                #print(len(decomposition.trend))
                df1 = pd.DataFrame({'trend':decomposition.trend})
                df1.index.freq = pd.infer_freq(df.index)
                trend1.append(df1)
                data_frames[i]['trend'] = decomposition.trend
                data_frames[i]['seasonal'] = decomposition.seasonal
                data_frames[i]['resid'] = decomposition.resid
            trend_p = []
            trend_p1 = {}
            next_index = trend1[0].index[-1]
            new_data_quarterlys = []
            new_dates = 0
            
            for i, df in enumerate(trend1):
                last_date = next_index.date()
                last_value = df['trend'].iloc[-5]
                new_dates = pd.date_range(start=last_date, periods=24, freq='QE')
                r = np.exp(np.log(1 + (last_value - df['trend'].iloc[-6]) / last_value) / 20)
                
                sequence = [last_value]
                for j in range(1, 24):
                    sequence.append(sequence[-1] * r)
                
                new_data_quarterly = pd.DataFrame({'date': new_dates, 'trend': sequence})
                new_data_quarterlys.append(new_data_quarterly)
            
            for k, df in enumerate(trend1):
                j = 0
                for i in range(len(df)):
                    if pd.isna(df['trend'].iloc[i]) and i >= len(df) / 2:
                        trend1[k].iloc[i, df.columns.get_loc('trend')] = new_data_quarterlys[k].iloc[j, new_data_quarterlys[k].columns.get_loc('trend')]
                        j += 1
            
            for i, df in enumerate(trend1):
                trend1[i]['Trend_lagged'] = df['trend'].shift(1)
            
            trend_models = self.fetch_linearmodels()
            for i, df in enumerate(trend1):
                df = df.fillna(0)
                trend_values = [0]
                
                for j in range(1, len(df)):
                    trend_values.append(df['Trend_lagged'].iloc[j] - df['Trend_lagged'].iloc[j - 1])
                
                df['trend_value'] = trend_values
                X = df[['Trend_lagged', 'trend_value']]
                y = df['trend']
                X = sm.add_constant(X)
                
                name1 = data_frames[i]['Value'].iloc[0]
                print(name1,trend_models[name1],type(trend_models),type(trend_models[name1]))
                results = trend_models[name1] 
                
                forecast_years = new_dates[4:]
                last_trend_value = df['trend'].iloc[-1]
                last_trend = df['trend_value'].iloc[-1]
                forecasts = []
                
                for year in forecast_years:
                    new_data = pd.DataFrame({'const': 1, 'Trend_lagged': [last_trend_value], 'trend_value': [last_trend]})
                    forecast = results.predict(new_data)[0]
                    forecasts.append(forecast)
                    last_trend = forecast - last_trend_value
                    last_trend_value = forecast
                    
                trend_p.append(forecasts)
                trend_p1[data_frames[i]['Value'].iloc[0]] = forecasts
                # print(trend_p1[data_frames[i]['Value'].iloc[0]])
                forecast_df = pd.DataFrame({'Year': forecast_years, 'Forecasted_Trend': forecasts})
                forecast_df.set_index('Year', inplace=True)
            next_index = data_frames[0].index[len(data_frames[0])-1]
            #print(next_index.date())
            ldate = next_index.date()
            trend_dataframe = []
            for i, df in enumerate(data_frames):
            # Generate 8 quarterly dates (2 years of data)
                last_date = ldate
                last_value = df['trend'].iloc[len(data_frames[i])-5]
                new_dates = pd.date_range(start=last_date, periods=24, freq='QE')
                r = np.exp(np.log(1 + (last_value- df['trend'].iloc[len(data_frames[i])-6]) / last_value) / 20)
                
                # Generate the sequence
                sequence = [last_value]
                for j in range(1, 24):
                    sequence.append(sequence[-1] * r)
                #print(len(sequence), sequence)
                # Generate 8 values following a logarithmic graph
                x = np.linspace(1, 500, 24)  # 1 to 9 to avoid log(0) issue
                log_values = np.log(x)
                # #print(log_values)
                # Normalize log values to start from the last value
                log_values_normalized = log_values - log_values[0] + last_value
                #print(len(log_values_normalized), last_value, len(data_frames[i]), df['trend'].iloc[len(data_frames[i])-7])
                # Create a new DataFrame for the generated values
                new_data_quarterly = pd.DataFrame({'date': new_dates, 'value': log_values_normalized, 'trend': sequence})
                #display(new_data_quarterly)
                # Plot the values
            
                # plt.figure(figsize=(10, 6))
                # plt.plot(new_data_quarterly['date'], new_data_quarterly['value'], marker='o', linestyle='-', color='b', label='Logarithmic Trend (Quarterly)')
                # plt.plot(new_data_quarterly['date'], new_data_quarterly['trend'], marker='o', linestyle='-', color='r', label='Logarithmic Trend (Quarterly)1 other')    
                
                # plt.xlabel('Date')
                # plt.ylabel('Value')
                # plt.title('Logarithmic Trend Values (Quarterly)'+ df['Value'].iloc[0])
                # plt.legend(new_data_quarterly['date'])
                new_data_quarterly['date'] = pd.to_datetime(new_data_quarterly['date'])
                new_data_quarterly.set_index('date', inplace=True)
                
                # Infer and set the frequency of the index
                new_data_quarterly.index.freq = pd.infer_freq(new_data_quarterly.index)
                
                trend_dataframe.append(new_data_quarterly)
            for k, df in enumerate(data_frames):
                ind1 = 0
                j = 0
                for i in range(len(df)):
                    ind = df.index[i]
                    # print(df.index[i], df1['trend'].iloc[i], df2[''].iloc[j])
                    if pd.isna(df['trend'].iloc[i]) and i>= len(df) -6:
                        ind1 += 1
                        data_frames[k].iloc[i, data_frames[k].columns.get_loc('trend')] = trend_dataframe[k].iloc[j, trend_dataframe[k].columns.get_loc('trend')]
                        #display(data_frames[k])
                        #display(trend_dataframe[k])
                        j +=1
                        #print(ind1)
                # df2[
                #print(ind1)
                subset_df2 = trend_dataframe[k].iloc[ind1:, [trend_dataframe[k].columns.get_loc('trend')]]
                #display(subset_df2)
                data_frames[k] = pd.concat([data_frames[k], subset_df2], axis = 0)
                #display(data_frames[k])
            self.trends = trend_p1
            # print(type(self.trends),type(trend_p1), self.trends, trend_p1)
            return data_frames, trend_p1
        except:
            raise ValueError(f"Data for '{self.company_name}' is not available.")
            
    def fetch_xgboostmodels(self):
        try:
            model_names = [
                "DandA", "Gross_Profit", "Revenue", "Operating_Expense", "Interest_Expense",
                 "SGandA", "Total_Current_Assets",
                "Capital_Expenditures", "Current_Ratio", "Gross_Margin", "Asset_Turnover", "DbR"
            ]
            # Define the company name
            company_name = self.company_name
            directory = '/Users/joshna/Desktop/Metis/Models/'
            # Initialize an empty dictionary to store the models
            models_dict = {}
            
            # Load each model and store it in the dictionary
            for name in model_names:
                filename = f"{directory}{company_name}{name}.json"
                # print(os.path.exists(filename))
                with open(filename, 'rb') as file:
                    model = xgb.XGBRegressor()
                    model.load_model(filename)
                    models_dict[name] = model
                    

            self.xgboost_models = models_dict
            # print(self.xgboost_models)
            return self.xgboost_models
        except:
            raise ValueError(f"Data for '{self.company_name}' is not available.")
            

    def fetch_linearmodels(self):
        try:
            model_names = [
                "DandA", "Gross_Profit", "Revenue", "Operating_Expense", "Interest_Expense",
                "EBIT", "EBITDA", "SGandA", "Total_Current_Liabilities", "Total_Current_Assets",
                "Capital_Expenditures", "Current_Ratio", "Gross_Margin", "Asset_Turnover", "DbR"
            ]
            
            # Define the company name
            company_name = self.company_name
            directory = '/Users/joshna/Desktop/Metis/Models/'
            # Initialize an empty dictionary to store the models
            models_dict = {}
            
            # Load each model and store it in the dictionary
            for name in model_names:
                filename = f"{directory}{company_name} trend {name} pipeline.pkl"
                with open(filename, 'rb') as file:
                    models_dict[name] = pickle.load(file)
                    #print("pickle models",models_dict[name])
            self.trend_pipelines = models_dict
            #print("pickle models",models_dict)
            return self.trend_pipelines
        except:
            raise ValueError(f"Data for '{self.company_name}' is not available.")
            
    def forecast_future_revenue(self, model, X_train, y_train, X, n_periods, k):
        try:
            future_preds = []
            current_X = y_train.iloc[0].values.reshape(1, -1)
        
            print(current_X, X_train[0], X.iloc[0,0])
            for i in range(1,n_periods):
                try:
                    pred = model.predict(current_X)
                except ValueError as e:
                    pred = model.predict([[current_X[0][1]]])
                    
                future_preds.append(pred[0])
        
                # Update current_X for the next prediction
                # values_to_append = np.array([ X_train[i-1], pred[0], X.iloc[i-1,2]])
        
                # Step 2: Append these values into current_X
                if k == 0:
                    current_X = np.append([X_train[i-1]], np.append( pred, X.iloc[i-1,2])).reshape(1, -1)
                # current_X =  pred.reshape(1, -1)
        
                # current_X = np.append(X.iloc[i,0],X_train[i-1], pred).reshape(1, -1)
                else:
                    current_X = np.append(y_train.iloc[i:i+1, 0:1], pred).reshape(1, -1)
                # print(current_X[:, 1:], current_X, pred, X_train[i-1])    
        
            return future_preds
        except:
            raise ValueError(f"Data for '{self.company_name}' is not available.")
    def get_predictions(self):
        try:
            if self.xgboost_models is None:
                self.xgboost_models = self.fetch_xgboostmodels()
            xgboostmodels = self.xgboost_models
            # trends = self.trends
            # print(trends)
            test_train = self.training_data
            xgboost_predictions = {}
            i = 0
            n_periods =21
            # print(test_train)
            for name, model in xgboostmodels.items() :
                trend_p = test_train[name][-1]
                xtest3 = test_train[name][2] 
                xtrain3 = test_train[name][0]
                # display(xtest3)
                print(xtrain3,type(test_train[name][2]), type(test_train[name][-1]) ,name)

            for name, model in xgboostmodels.items() :
                trend_p = test_train[name][-1]
                xtest3 = test_train[name][2] 
                xtrain3 = test_train[name][0]
                if "Asset_Turnover_lag_1" in xtest3.columns and name == "Revenue":
                    xtest3.drop(columns=["Asset_Turnover_lag_1"],inplace=True)
                # display(xtest3)
                
                # print(name,trend_p, type(trend_p), test_train[name][-1])
                if name == "Total_Current_Assets":
                    print(name, "prediction for xgboost")
                    future_revenue_preds = self.forecast_future_revenue(model, trend_p, xtest3, xtrain3, n_periods,  0)
                    print(name, "prediction done for xgboost")
                    xgboost_predictions[name] = future_revenue_preds
                else:
                    print(name, "prediction for xgboost")
                    future_revenue_preds = self.forecast_future_revenue(model, trend_p, xtest3, xtrain3, n_periods,  1)
                    print(name, "prediction done for xgboost")
                    xgboost_predictions[name] = future_revenue_preds
            xgboost_predictions["Total_Current_Liabilities"] = [i/j for i,j in zip(xgboost_predictions["Total_Current_Assets"],xgboost_predictions["Current_Ratio"])]
            xgboost_predictions["EBIT"] = [i-j for i,j in zip(xgboost_predictions["Revenue"],xgboost_predictions["DandA"])]
            
            xgboost_predictions["NWC"] = [i-j for i,j in zip(xgboost_predictions["Total_Current_Assets"],xgboost_predictions["Total_Current_Liabilities"])]
            print(xgboost_predictions["Total_Current_Liabilities"], xgboost_predictions["Total_Current_Assets"])
            xgboost_predictions["Taxes"] = [0.21 for i in range(len(xgboost_predictions["Total_Current_Assets"]))]
            for key, v in xgboost_predictions.items():
                print(key, v)
            print(xgboost_predictions.keys())
            self.xgboost_predictions = xgboost_predictions
            
        except:
            raise ValueError(f"Data for '{self.company_name}' is not available.")
            
    def get_final_predictions(self):
        try:
            if self.xgboost_predictions is None:
                self.get_predictions()
            trends = self.trends
            # print(trends)

            predictions = self.xgboost_predictions
            print(self.xgboost_predictions)
            print(self.xgboost_predictions.keys())
            final_predictions = {}
            for name, trend in trends.items():
                if name in predictions.keys():
                    n1 = [(i*0.3)+(j) for i, j in zip(trend, predictions[name])]
                    final_predictions[name] = n1
            final_predictions["Total_Current_Liabilities"] = [i/j for i,j in zip(final_predictions["Total_Current_Assets"],final_predictions["Current_Ratio"])]
            final_predictions["EBIT"] = [i-j for i,j in zip(final_predictions["Revenue"],final_predictions["DandA"])]
            
            final_predictions["NWC"] = [i-j for i,j in zip(final_predictions["Total_Current_Assets"],final_predictions["Total_Current_Liabilities"])]
            final_predictions["Taxes"] = [0.21 for i in range(len(final_predictions["Total_Current_Assets"]))]
            self.final_predictions = final_predictions
        except:
            raise ValueError(f"Data for '{self.company_name}' is not available.")
            
    def calculate_fcf(self):
        if self.prepped_data is None:
            self.prep_data(self.historicals)
        if self.final_predictions is None:
            self.get_final_predictions()
        
        operating_income = self.final_predictions['Revenue']
        
        ebit = self.final_predictions['EBIT']
        taxes = self.final_predictions['Taxes']
        depreciation = self.final_predictions['DandA']
        capex = self.final_predictions['Capital_Expenditures']
        change_in_nwc = self.final_predictions['NWC']
        
        unlevered_free_cash_flows = []
        for i in range(len(operating_income)):
            print(i, "operating_income",operating_income[i], "DandA" , depreciation[i], "capx", capex[i], "Chnin_nwc",change_in_nwc[i],"Total_Current_Assets",self.final_predictions["Total_Current_Assets"][i],"Total_Current_Liabilities",self.final_predictions["Total_Current_Liabilities"][i],'\n') 
            fcf = (ebit[i]*(1-0.21)) + depreciation[i] - capex[i] - change_in_nwc[i]
            print(i, "operating_income",operating_income[i],"ebit",ebit[i], "DandA" , depreciation[i], "capx", capex[i], "Chnin_nwc",change_in_nwc[i],"fcf", fcf,'\n') 
            unlevered_free_cash_flows.append(fcf)
        
        self.ufcf = unlevered_free_cash_flows
        
        # return free_cash_flows
    
    def calculate_dcf(self):
        if self.fcf is None:
            self.calculate_fcf()
        if self.Wacc is None:
            self.calculate_wacc()
        
        ufree_cash_flows = self.ufcf  # Use stored FCF
        # terminal_value = self.data.get('terminal_value', 0)
        TGR = self.TGR
        print(self.Debt)
        Debt = self.Debt[0]
        Wacc = self.wacc
        Cash = self.Cash
        Shares = self.Shares
        present_fcf = [ufree_cash_flows[i]/ ((1+Wacc)**((i+1)/5)) for i in range(0,len(ufree_cash_flows))]
        print("ufree cash flows",present_fcf)
        # dcf_value = np.sum(free_cash_flows) + terminal_value / (1 + discount_rate)
         # p = ufcf/1+wacc^year
        terminalValue = ufree_cash_flows[-1]*(1+TGR)/(Wacc- TGR)
        present_terminalvalue = terminalValue/(1+Wacc)**5
        EnterpriseValue = sum(present_fcf)+ present_terminalvalue
        EquityValue = EnterpriseValue +Debt -Cash
        ImpliedSharePrice = EquityValue/ Shares
        
        self.terminalValue = terminalValue
        self.present_terminalvalue = present_terminalvalue
        self.EnterpriseValue = EnterpriseValue
        self.EquityValue = EquityValue
        self.ImpliedSharePrice = ImpliedSharePrice
        self.dcf = {"terminalValue":terminalValue, "present_terminalvalue":present_terminalvalue,"EnterpriseValue":EnterpriseValue,"EquityValue":EquityValue, "ImpliedSharePrice":ImpliedSharePrice}
        return self.dcf
    def update_values(self,analysis_type,  **kwargs):
        try:
            final_predictions = self.final_predictions
            key_items = next(iter(kwargs.keys()))
            # next(iter(kwargs.keys()))
            items = kwargs[key_items]
            
            print(kwargs.items(), type(kwargs), type(kwargs.items()), key_items, type(key_items), items, type(items))
            for i, (key, value) in enumerate(items.items()):
                if key in final_predictions:
                    # k = 0
                    print(key, value)
                    for j in range(0, len(final_predictions[key])):
                        # print("j", j, final_predictions[key][j])
                        # for i in range(len(financial_statement)):
                        if j >= 12 and j<16:
                            if "1" in value.keys(): 
                            # Apply growth rate x to the previous index (index 11)
                                growth_rate = 1+value["1"]
                                # print(growth_rate,final_predictions[key][j], final_predictions[key][j-1])
                                final_predictions[key][j] = final_predictions[key][j-1] * growth_rate
                                # print(growth_rate,final_predictions[key][j], final_predictions[key][j-1])
                        if j >= 16:
                            # Apply growth rate y to the previous index (index 15)
                            if "2" in value.keys():
                                growth_rate = 1+value["2"]
                                # print(growth_rate,final_predictions[key][j], final_predictions[key][j-1])
                                final_predictions[key][j] = final_predictions[key][j-1] * growth_rate
                                # print(growth_rate,final_predictions[key][j], final_predictions[key][j-1])
                        # if len(value)+j >= len(final_predictions[key]):
                        #     final_predictions[key][j] = value[j-k]
                        # else:
                        #     k = k + 1
                    print(key, final_predictions[key])
            if "Total_Current_Liabilities" not in items.keys():
                final_predictions["Total_Current_Liabilities"] = [i/j for i,j in zip(final_predictions["Total_Current_Assets"],final_predictions["Current_Ratio"])]
            if "EBIT" not in items.keys():
                final_predictions["EBIT"] = [i-j for i,j in zip(final_predictions["Revenue"],final_predictions["DandA"])]
            if "NWC" not in items.keys():
                final_predictions["NWC"] = [i-j for i,j in zip(final_predictions["Total_Current_Assets"],final_predictions["Total_Current_Liabilities"])]
            if "Taxes" not in items.keys():
                final_predictions["Taxes"] = [0.21 for i in range(len(final_predictions["Total_Current_Assets"]))]
            # self.final_predictions = final_predictions
            self.final_predictions = final_predictions
            if analysis_type == "DCF":
                self.calculate_fcf()
                self.calculate_dcf()
            elif analysis_type == "FCF":
                self.calculate_fcf()
            elif analysis_type == "WACC":
                self.calculate_wacc()
        except:
            raise ValueError(f"Data for '{self.company_name}' is not available.")
            
    def beta1(self, company_code):
        # try:
        # Retrieve stock beta
        if company_code[-2:] == ".O":
            company_code = company_code[:-2]
        stock = yf.Ticker(company_code)
        beta = stock.info['beta']
        if beta is not None:
            return beta
        else:
    # Download historical price data
            semiconductor_index = yf.download('^SOX', start='2020-01-01', end='2023-01-01')['Adj Close']
            market_index = yf.download('^GSPC', start='2020-01-01', end='2023-01-01')['Adj Close']
            
            # Calculate daily returns
            semiconductor_returns = semiconductor_index.pct_change().dropna()
            market_returns = market_index.pct_change().dropna()
            
            # Align the data
            data = pd.DataFrame({'Semiconductor': semiconductor_returns, 'Market': market_returns}).dropna()
            
            # Perform regression
            X = sm.add_constant(data['Market'])
            model = sm.OLS(data['Semiconductor'], X).fit()
            
            # Get the beta value
            beta = model.params['Market']
            return beta
    def calculate_wacc(self):
        Equity = self.Equity
        Debt = self.Debt
        Equity = Equity[0]
        Debt = Debt[0]
        interestexpense = self.interestexpense
        interestexpense = interestexpense[0]
        # Taxes1 = -sum(incometax[0])/(sum(ebit0[0]))
        Taxes = self.Taxes
        
        
        print(Equity, Debt, Taxes,interestexpense)
        
        EpD = Equity + Debt
        PoEquity = Equity/EpD
        PoDebt = 1 - PoEquity
        UtreBond = self.UtreBond
        # Beta = 1.39 /(1+((1-Taxes)*(Debt/Equity)))
        self.beta = self.beta1(self.company_code)
        beta = self.beta
        # print( Beta, EpD, PoEquity, PoDebt, beta)
        
        marketriskpremium = self.marketriskpremium
        CoE = UtreBond+beta*marketriskpremium
        CoD = (interestexpense/Debt)*(1-0.21)
        
        # print(CoE, CoD)
        
        Wacc = (PoEquity * CoE)+(PoDebt * CoD*(1-Taxes))
        print("Wacc", Wacc)
        # wacc = cost_of_equity * equity_weight + cost_of_debt * debt_weight
        # self.Wacc = Wacc
        Wacc_value = {"Equity":Equity, "Debt": Debt, "Percentage of Equity": PoEquity, "Percentage of Debt": PoDebt, "Beta":beta,"Cost of Equity": CoE, "Cost of Debt": CoD, "WACC":Wacc}
        self.Wacc = Wacc_value
        self.wacc=Wacc
        # return Wacc
    
    def monte_carlo_simulation(self):
        try:
            final_predictions = self.final_predictions
            results = {}
            for key, values in final_predictions.items():
                mean = np.mean(values)
                std_dev = np.std(values)
                results[key] = {'mean': mean, 'std_dev': std_dev}
            
            # Define function for Monte Carlo simulation
            def monte_carlo_simulation(mean, std_dev, num_simulations=1000):
                return np.random.normal(mean, std_dev, num_simulations)
            
            # Perform Monte Carlo simulations for each key
            num_simulations = 1000
            mc_results = {}
            for key, stats in results.items():
                mc_results[key] = monte_carlo_simulation(stats['mean'], stats['std_dev'], num_simulations)
            
            # # Print results
            # print("Mean and Standard Deviation for each key:")
            # for key, stats in results.items():
            #     print(f"{key}: Mean = {stats['mean']}, Std Dev = {stats['std_dev']}")
            
            # print("\nMonte Carlo Simulation Results (first 10 values for each key):")
            # for key, simulations in mc_results.items():
            #     print(f"{key}: {simulations[:10]}")
            # mean = self.data.get('mean', 0)
            # std_dev = self.data.get('std_dev', 1)
            # num_simulations = self.data.get('num_simulations', 1000)
            
            # simulations = np.random.normal(mean, std_dev, num_simulations)
            mc_result = {}
            for k, v in mc_results.items():
                mc_result[k] = {"Max": max(v), "Min": min(v)}
            self.monte_carlo_simulations = mc_result
            return self.monte_carlo_simulations
        except:
            raise ValueError(f"Data for '{self.company_name}' is not available.")
        
    def get_latest_share_price(self,ticker):
        if ticker[-2:] == ".O":
            ticker = ticker[:-2]
        stock = yf.Ticker(ticker)
        latest_price = stock.history(period='1d')['Close'].iloc[-1]
        return latest_price
        
    def cca_get_companies(self, companies):
        try:
                
            def extract_data_between_parentheses(text):
                # Define the regular expression pattern to match text between parentheses
                sheets_data = self.sheets_data
                pattern = r'\((.*?)\)'
                
                # Use re.findall to find all matches of the pattern in the text
                matches = re.findall(pattern, text)
                
                return matches
            files_path = r"/Users/joshna/Downloads/fwddatabasestuff"
            companies_data = {}
            for company in companies:
                full_path = os.path.join(files_path, f"{company} Financials.xlsx")
                file_path = r""+full_path
                if os.path.exists(file_path):
                    print("A")
                else:
                    print("B",file_path,"C",full_path)

                excel_workbook = pd.ExcelFile(file_path)
                # Dictionary to hold data from each sheet
                sheets_data = {}
                for sheet_name in excel_workbook.sheet_names:
                    sheet_data = pd.read_excel(file_path, sheet_name=sheet_name)
                    sheets_data[sheet_name] = sheet_data
                # self.sheets_data = sheets_data
                companies_data[company] = sheets_data
            self.cca_historicals_data = companies_data
            companies_code_name = {}
            for company in companies:
                company_name = ""
                company_code = ""
                sheets_data = self.sheets_data
                sheets_data = companies_data[company]
                for sheet_name, data in sheets_data.items():
                    if sheet_name == 'Income Statement':
                        company_name = data[data.iloc[:, 0] == "Company Name"].iloc[:, 1:2].values
                print(company_name[0][0])
                company_name = company_name[0][0]
                company_code = extract_data_between_parentheses(company_name)[0]
                companies_code_name[company_code] = company_name
            self.cca_company_names_codes = companies_code_name
            print(companies_data.keys(),companies_data["Intel Corp"].keys())
            return companies_code_name
        except:
            raise ValueError(f"Data for is not available.")
        
    def cca_get_historicals(self, companies_name):
        try:
            if self.cca_company_names_codes is None:
                self.cca_get_companies(companies_name)
            cca_companies = self.cca_company_names_codes
            companies_data = {}
            companies_h_data = self.cca_historicals_data
            # dates = 0
            # dates = dates[0]
            for (company_code,company_name),(companies, company_data) in zip(cca_companies.items(),companies_h_data.items()):
                items = {}
                for sheet_name, data in company_data.items():
                    Share_price = self.get_latest_share_price(company_code)
                    items["Share_price"]=Share_price
                    if sheet_name == 'Income Statement':
                        revenue_data = data[data.iloc[:, 0] == "Revenue from Business Activities - Total"].iloc[:, 1:2].values
                        revenue = revenue_data[~np.isnan(revenue_data.astype(float))].tolist()
                        
                        if revenue:
                            items["Revenue"] =  revenue[0]
                        
                        
                        ebitda_data = data[data.iloc[:, 0] == "Earnings before Interest, Taxes, Depreciation & Amortization (EBITDA)"].iloc[:, 1:2].values
                        ebitda = ebitda_data[~np.isnan(ebitda_data.astype(float))].tolist()
                        if ebitda:
                            items["EBITDA"]= ebitda[0]
                        Income = data[data.iloc[:, 0] == "Income Available to Common Shares"].iloc[:, 1:2].values
                        Income = Income[0]
                        items["Income"]= Income[0]

                    elif sheet_name == 'Financial Summary':
                        Shares = data[data.iloc[:, 0] == "Common Shares - Outstanding - Total"].iloc[:, 1:2].values
                        Shares = Shares[0]
                        items["Shares"] =  Shares[0]
                    elif sheet_name == 'Balance Sheet':
                        Equity = data[data.iloc[:, 0] == "Total Shareholders' Equity - including Minority Interest & Hybrid Debt"].iloc[:, 1:5].values
                        Debt = data[data.iloc[:, 0] == "Debt - Total"].iloc[:, 1:5].values
                        Equity = Equity[0]
                        
                        items["Equity"] = Equity[0]
                        Debt = Debt[0]
                        items["Debt"] =  Debt[0]
                        EnterpriseValue = Equity[0] +Debt[0]
                        items["EnterpriseValue"] = EnterpriseValue
                companies_data[companies] = items
            print(companies_data)
            self.cca_data = companies_data
            return companies_data
        except:
            raise ValueError(f"Data for is not available.")
    def cca(self):
        try:
            
            # if self.cca_data is None:
            #    self.cca_get_historicals(companies_name)
            cca_data = self.cca_data
            data_for_comparison = {}
            EVbR = []
            EVbE = []
            PbE = []
            for company_code, company_data in cca_data.items():
                revenue = company_data["Revenue"]
                EnterpriseValue = company_data["EnterpriseValue"]
                EVbR.append(EnterpriseValue/revenue)
                Ebitda = company_data["EBITDA"]
                EVbE.append(EnterpriseValue/Ebitda)
                EquityValue = company_data["Equity"]
                Income = company_data["Income"]
                PbE.append(EquityValue/Income)
                cca_data[company_code]["EV/Revenue"] = EVbR
                cca_data[company_code]["EV/EBITDA"] = EVbE
                cca_data[company_code]["P/E"] = PbE
            print (cca_data)
            comp_avg = {}
            for par, va in zip(["EV/Revenue", "EV/EBITDA", "P/E"], [EVbR, EVbE, PbE]):
                High = max(va[1:])
                p75 = np.percentile(va[1:], 75)
                Avg = np.mean(va[1:])
                Med = np.median(va[1:])
                p25 = np.percentile(va[1:], 25)
                Low = min(va[1:])
                comp_avg[par] = {"High": High, "Percentile 75": p75, "Average": Avg, "Median": Med,"Percentile 25": p25, "Low": Low}
            evaluation = {}
            print('\n',comp_avg)
            first_index=next(iter(cca_data.keys()))
            for par in ["EV/Revenue", "EV/EBITDA", "P/E"]:
                
                ImpliedEnterpriseValue = comp_avg[par]["Median"] * cca_data[first_index]["Revenue"]
                NetDebt = cca_data[first_index]["Debt"]
                ImpliedMarketValue = ImpliedEnterpriseValue-NetDebt
                SharesOutstanding = cca_data[first_index]["Shares"]
                ImpliedValuePerShare = ImpliedMarketValue/SharesOutstanding
               
                evaluation[par] = {"Implied Enterprise Value": ImpliedEnterpriseValue, "Net Debt": NetDebt, "Implied Market Value": ImpliedMarketValue, "Shares Outstanding": SharesOutstanding, "Implied Value per Share": ImpliedValuePerShare}
            print(evaluation)
            self.cca_data = cca_data
            self.comp_avg = comp_avg
            self.evaluation = evaluation
        except:
            raise ValueError(f"Data for is not available.")

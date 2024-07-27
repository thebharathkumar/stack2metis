import spacy
from ForecastingEngine import FinancialModeling

nlp = spacy.load('en_core_web_sm')

def simulate_llm_response(prompt):
    return prompt

def parse_request(request):
    analysis_type = None
    companies = []
    historical = False

    parts = [part.strip() for part in request.split(',')]
    
    if len(parts) < 2:
        return None, None, None, None
    
    company_part = parts[0]
    analysis_part = parts[1]
    
    if "DCF" in analysis_part.upper():
        analysis_type = "DCF"
    elif "FCF" in analysis_part.upper():
        analysis_type = "FCF"
    elif "WACC" in analysis_part.upper():
        analysis_type = "WACC"
    elif "CCA" in analysis_part.upper():
        analysis_type = "CCA"
    elif "HISTORICAL" in analysis_part.upper():
        analysis_type = "HISTORICAL"
        historical = True

    doc = nlp(company_part)
    org_entities = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    
    companies.extend(org_entities)
    
    if len(companies) == 0:
        return None, None, None, None, None

    print("Companies:", companies)
    print("Analysis Type:", analysis_type)
    
    return companies, analysis_type, historical

def run_forecasting_engine(analysis_type, companies, growth_rate=None, line_item=None):
    result = {}
    print(type(companies), type(analysis_type)) 
    forecastengineresults = FinancialModeling(analysis_type, companies)
    
    if analysis_type == "DCF":
        DCF, FCF, MonteCarloSimulations = forecastengineresults.return_dcf_and_fcf_montecarlo()
        result['DCF'] = DCF
        result['FCF'] = FCF
        result['MonteCarloSimulations'] = MonteCarloSimulations
    elif analysis_type == "FCF":
        FCF, MonteCarloSimulations = forecastengineresults.return_fcf_montecarlo()
        result['FCF'] = FCF
        result['MonteCarloSimulations'] = MonteCarloSimulations
    elif analysis_type == "WACC":
        result['WACC'] = forecastengineresults.return_wacc()
    elif analysis_type == "CCA":
        if len(companies) > 1:
            forecastengineresults.cca_company_name_B = companies[1]
            CCA_data, Comp_Avg, CCA = forecastengineresults.return_cca()
            result['CCA_data'] = CCA_data
            result['Comp_Avg'] = Comp_Avg
            result['CCA'] = CCA
        else:
            result = "To run CCA analysis, two companies are required."
    elif analysis_type == "HISTORICAL":
        result['Historical Data'] = forecastengineresults.return_historicals()
    else:
        result = "Unsupported analysis type."

    if isinstance(result, dict) and growth_rate is not None:  
        result['Growth Rate'] = f"{growth_rate}%"  
    return result

def format_results(results):
    if isinstance(results, dict):
        formatted = "The results of your analysis are:\n"
        for key, value in results.items():
            if isinstance(value, dict):
                formatted += f"{key}:\n"
                for year, data in value.items():
                    formatted += f"  {year}: {data}\n"
            else:
                formatted += f"{key}: {value}\n"
        return formatted
    else:
        return results

previous_results = []

def store_results(results):
    previous_results.append(results)

def generate_dict_for_update(line_item, growth_rate1, growth_rate2):
    items = ['Revenue', 'EBIT', 'DandA', 'Taxes', 'Capital_Expenditures', 'NWC']
    update = {}
    if line_item in items:
        update[line_item] = [growth_rate1, growth_rate2]
    elif line_item is None:
        for item in items:
            update[item] = [growth_rate1, growth_rate2]
    else:
        print(f"Line item '{line_item}' not recognized. Skipping update for this item.")
    return update

def ask_for_growth_rate_and_line_item():
    while True:
        growth_rate_input = input("Enter the growth rate (in %) for two years, and specify a line item if any (e.g., '11 5 Revenue'): ")
        try:
            parts = growth_rate_input.split()
            growth_rate1 = float(parts[0])
            growth_rate2 = float(parts[1])
            if abs(growth_rate1) > 1: 
                growth_rate1 /= 100  
            if abs(growth_rate2) > 1:
                growth_rate2 /= 100
            if len(parts) > 2:
                line_item = ' '.join(parts[2:])
            else:
                line_item = None
            update = generate_dict_for_update(line_item, growth_rate1, growth_rate2)
            return growth_rate1, growth_rate2, line_item
        except ValueError:
            print("Invalid input. Please enter a valid growth rate.")

def process_user_request(prompt):
    request = simulate_llm_response(prompt)
    companies, analysis_type, historical = parse_request(request)
    if analysis_type and companies:
        return companies, analysis_type, historical
    else:
        print("Sorry, I couldn't understand your request. Please specify a valid financial analysis with required details.")
        return None, None, None, None

def main():
    while True:
        user_input = input("Please enter your request (format: 'Company Name, Analysis Type' or 'exit' to quit): ").strip()
        if user_input.lower() == "exit":
            print("Exiting the program.")
            break
        companies, analysis_type, historical = process_user_request(user_input)
        if analysis_type and companies:
            if historical:
                print(f"Fetching historical data for {companies[0]}...")
                results = run_forecasting_engine(analysis_type, companies)
            else:
                print(f"Running {analysis_type} analysis for {', '.join(companies)}...")
                results = run_forecasting_engine(analysis_type, companies)

            while True:
                continue_input = input("Do you want to continue (yes/no)? ").strip().lower()
                if continue_input == "no":
                    print("Exiting the program.")
                    break
                elif continue_input == "yes":
                    if analysis_type in ["DCF", "FCF"]:
                        growth_rate1, growth_rate2, line_item = ask_for_growth_rate_and_line_item()
                    else:
                        growth_rate1 = None
                        growth_rate2 = None
                        line_item = None
                    results = run_forecasting_engine(analysis_type, companies, [growth_rate1, growth_rate2], line_item)
                    formatted_results = format_results(results)
                    store_results(formatted_results)
                    print(formatted_results)
                    break
                else:
                    print("Invalid input. Please enter 'yes' or 'no'.")
        else:
            print("Sorry, I couldn't understand your request. Please specify a valid financial analysis with required details.")

if __name__ == "__main__":
    main()
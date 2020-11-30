import pandas as pd

def serialize(dataFrame, column):
    return [x for x in range(len(dataFrame[column].unique()))]

def replaceDict(dataFrame, column):
    vals = serialize(dataFrame, column)
    return dict(zip(dataFrame[column].unique(), vals))

UNREST_COLUMNS = ["EVENT_ID_CNTY", 
                    "EVENT_DATE", 
                    "EVENT_TYPE", 
                    "REGION", 
                "FATALITIES"]

CASES_COLUMNS = ["iso_code",
                "continent",
                "location", 
                "date", 
                "total_cases", 
                "new_cases", 
                "total_deaths", 
                "reproduction_rate", 
                "hosp_patients", 
                "positive_rate", 
                "stringency_index", 
                "population",
                "median_age",
                "gdp_per_capita",
                "life_expectancy"
            ]

unrest_df = pd.read_csv("./coronavirus_Oct31.csv")
covid_cases_df = pd.read_csv("./owid-covid-data.csv")

unrest_df = unrest_df[unrest_df.columns.intersection(UNREST_COLUMNS)]
covid_cases_df = covid_cases_df[covid_cases_df.columns.intersection(CASES_COLUMNS)]


unrest_afg = unrest_df[unrest_df["EVENT_ID_CNTY"].str.contains("AFG")]
cases_afg = covid_cases_df.query("iso_code == 'AFG'")

unrest_afg.EVENT_DATE = pd.to_datetime(unrest_afg.EVENT_DATE)
cases_afg.date = pd.to_datetime(cases_afg.date)

merge = unrest_afg.merge(cases_afg, how="inner", left_on="EVENT_DATE", right_on="date")

classification = serialize(unrest_df, "EVENT_TYPE")

issueType = unrest_afg['EVENT_TYPE']
issueType = issueType.replace(replaceDict(unrest_df, "EVENT_TYPE"))

# print(issueType)

# print(classification)

# print(replaceDict(unrest_df, "EVENT_TYPE"))

merge = merge.drop('EVENT_TYPE', axis=1)

print(merge)


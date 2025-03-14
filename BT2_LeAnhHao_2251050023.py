# Author: Le Anh Hao
# MSSV: 2251050023

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ===================== TEST ================================


# ===================== PANDAS ===============================
def read_file():
    return pd.read_csv("04_gap-merged.tsv", sep="\t")


# Bai 1
def bai_1_in_5_dong_dau():
    print(">>>Bai 1")
    df = read_file()
    print(df.head())


# Bai 2
def bai_2_so_dong_so_cot():
    print(">>>Bai 2")
    df = read_file()
    row, col = df.shape
    print(f"Số dòng: {row} \nSố cột: {col}")


# Bai 3
def bai_3_in_ten_cot():
    print(">>>Bai 3")
    df = read_file()
    print(df.columns.to_list())


# Bai 4
def bai_4_type_of_the_column_name():
    print(">>>Bai 4")
    df = read_file()
    for col in df.columns:
        print(f"{col}: {type(col)}")


# Bai 5
def bai_5_lay_cot_country():
    print(">>>Bai 5")
    df = read_file()
    # # Trả về Series, giữ nguyên chỉ mục cũ
    # country = df['country'].drop_duplicates()

    # Trả về Series, đặt lại chỉ mục
    # country = df['country'].drop_duplicates().reset_index(drop=True)

    country = df["country"]
    print(country.head())


# Bai 6
def bai_6_lay_5_country_cuoi():
    print(">>>Bai 6")
    df = read_file()
    country = df["country"]
    print(country.tail())


# Bai 7
def bai_7_lay_country_continent_year():
    print(">>>Bai 7")
    df = read_file()
    print("First 5 observations")
    print(df[["country", "continent", "year"]].head())
    print("Last 5 observations")
    print(df[["country", "continent", "year"]].tail())


# ==================== CHAT GPT GIAI =======================
# Bai 8: Lấy dòng đầu tiên và dòng thứ 100
def bai_8_get_first_and_100th_row():
    df = read_file()
    print("First row:")
    print(df.iloc[0])
    print("\n100th row:")
    print(df.iloc[99])


# Bai 9: Lấy cột đầu tiên và cả cột đầu + cột cuối
def bai_9_get_first_and_last_column():
    df = read_file()
    print("First column:")
    print(df.iloc[:, 0])
    print("\nFirst and last column:")
    print(df.iloc[:, [0, -1]])


# Bai 10: Lấy dòng cuối cùng với .loc
def bai_10_get_last_row():
    df = read_file()
    print("Last row using .loc:")
    print(df.loc[df.index[-1]])

    print("\nTrying index -1 with .loc (should fail):")
    try:
        print(df.loc[-1])
    except KeyError:
        print("Index -1 is not valid with .loc")


# Bai 11: Chọn dòng đầu tiên, dòng thứ 100 và dòng thứ 1000
def bai_11_select_specific_rows():
    df = read_file()
    print("Using .iloc:")
    print(df.iloc[[0, 99, 999]])
    print("\nUsing .loc:")
    print(df.loc[[0, 99, 999]])


# Bai 12: Lấy quốc gia thứ 43 với .loc và .iloc
def bai_12_get_43rd_country():
    df = read_file()
    print("Using .iloc:")
    print(df.iloc[42]["country"])
    print("\nUsing .loc:")
    print(df.loc[42, "country"])


# Bai 13: Lấy dòng 1, 100, 1000 từ cột 1, 4, 6
def bai_13_get_specific_rows_columns():
    df = read_file()
    print(df.iloc[[0, 99, 999], [0, 3, 5]])


# Bai 14: Lấy 10 dòng đầu tiên
def bai_14_get_first_10_rows():
    df = read_file()
    print(df.head(10))


# Bai 15: Tính tuổi thọ trung bình theo năm
def bai_15_avg_life_expectation():
    df = read_file()
    print(df.groupby("year")["lifeExp"].mean())


# Bai 16: Dùng phương pháp subsetting để giải bài 15
def bai_16_avg_life_expectation_subset():
    df = read_file()
    years = df["year"].unique()
    for year in years:
        avg_life = df[df["year"] == year]["lifeExp"].mean()
        print(f"Year {year}: {avg_life}")


# Bai 17: Tạo Series với index 0 là 'banana' và index 1 là '42'
def bai_17_create_simple_series():
    s = pd.Series(["banana", 42], index=[0, 1])
    print(s)


# Bai 18: Tạo Series với index 'Person' và 'Who'
def bai_18_create_named_series():
    s = pd.Series(["Wes McKinney", "Creator of Pandas"], index=["Person", "Who"])
    print(s)


# Bai 19: Tạo DataFrame từ dictionary
def bai_19_create_pandas_dict():
    data = {
        "Occupation": ["Chemist", "Statistician"],
        "Born": ["1920-07-25", "1876-06-13"],
        "Died": ["1958-04-16", "1937-10-16"],
        "Age": [37, 61],
    }
    df = pd.DataFrame(data, index=["Franklin", "Gosset"])
    print(df)


# ===================== BAI GIAI ===========================
# bai_1_in_5_dong_dau()
# bai_2_so_dong_so_cot()
# bai_3_in_ten_cot()
# bai_4_type_of_the_column_name()
# bai_5_lay_cot_country()
# bai_6_lay_5_country_cuoi()
# bai_7_lay_country_continent_year()

# bai_8_get_first_and_100th_row()
# bai_9_get_first_and_last_column()
# bai_10_get_last_row()
# bai_11_select_specific_rows()
# bai_12_get_43rd_country()
# bai_13_get_specific_rows_columns()
# bai_14_get_first_10_rows()
# bai_15_avg_life_expectation()
# bai_16_avg_life_expectation_subset()
# bai_17_create_simple_series()
# bai_18_create_named_series()
# bai_19_create_pandas_dict()


# ======================= PANDAS & SEABORN =========================
data = [4879, 12104, 12756, 6792, 142984, 120536, 51118, 49528]
series = pd.Series(
    data,
    index=[
        "Mercury",
        "Venus",
        "Earth",
        "Mars",
        "Jupyter",
        "Saturn",
        "Uranus",
        "Neptune",
    ],
)


# Bai 1
def bai_1():
    series = pd.Series(data)
    print(series)


# Bai 2
def bai_2():
    print(series)


# Bai 3
def bai_3():
    print(f"Duong kinh Trai Dat: {series['Earth']}")


# Bai 4
def bai_4():
    print(series["Mercury":"Mars"])


# Bai 5
def bai_5():
    print(series[["Earth", "Jupyter", "Neptune"]])


# Bai 6
def bai_6():
    series["Pluto"] = 2370
    print(series)


# Tao data planets
data = {
    "diameter": [4879, 12104, 12756, 6792, 142984, 120536, 51118, 49528, 2370],
    "avg_temp": [167, 464, 15, -65, -110, -140, -195, -200, -225],
    "gravity": [3.7, 8.9, 9.8, 3.7, 23.1, 9.0, 8.7, 11.0, 0.7],
}
planets = pd.DataFrame(data)


# Bai 7
def bai_7():
    print(planets)


# Bai 8
def bai_8():
    print(planets.head(3))


# Bai 9
def bai_9():
    print(planets.tail(2))


# Bai 10
def bai_10():
    print(planets.columns)


# Bai 11
def bai_11():
    planets.index = [
        "Mercury",
        "Venus",
        "Earth",
        "Mars",
        "Jupyter",
        "Saturn",
        "Uranus",
        "Neptune",
        "Pluto",
    ]
    print(planets)


# Bai 12
def bai_12():
    print(planets["gravity"])


# Bai 13
def bai_13():
    print(planets[["gravity", "diameter"]])


# Bai 14
def bai_14():
    print(planets.loc["Earth"], ["gravity"])


# Bai 15
def bai_15():
    print(planets.loc[["Earth"], ["gravity", "diameter"]])


# Bai 16
def bai_16():
    print(planets.loc["Earth":"Saturn", ["gravity", "diameter"]])


# Bai 17
def bai_17():
    print(planets["diameter"] > 1000)


# Bai 18
def bai_18():
    print(planets[planets["diameter"] > 10000])


# Bai 19
def bai_19():
    print(planets[(planets["avg_temp"] > 0) & (planets["gravity"] > 5)])


# Bai 20
def bai_20():
    print(planets["diameter"].sort_values())


# Bai 21
def bai_21():
    print(planets["diameter"].sort_values(ascending=False))


# Bai 22
def bai_22():
    print(planets.sort_values(by="gravity", ascending=False))


# Bai 23
def bai_23():
    print(planets.loc["Mercury"].sort_values())


# ====================== SEABORNS ====================================
# Bai 1
tips = sns.load_dataset("tips")
sns.set_style("whitegrid")
g = sns.lmplot(x="tip",
y="total_bill",
data=tips,
aspect=2)
g = (g.set_axis_labels("Tip","Total bill(USD)").
set(xlim=(0,10),ylim=(0,100)))
plt.title("title")
plt.show()


# Bai 2
def bai_2_seaborn():
    print(sns.get_dataset_names())


# ====================== RUN =========================================
# bai_1()
# bai_2()
# bai_3()
# bai_4()
# bai_5()
# bai_6()
# bai_7()
# bai_8()
# bai_9()
# bai_10()
# bai_11()
# bai_12()
# bai_13()
# bai_14()
# bai_15()
# bai_16()
# bai_17()
# bai_18()
# bai_19()
# bai_20()
# bai_21()
# bai_22()
# bai_23()

bai_2_seaborn()

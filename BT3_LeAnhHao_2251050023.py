from sklearn import neighbors, datasets, preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.datasets import load_breast_cancer, load_diabetes, fetch_lfw_people
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
import mglearn
from sklearn.svm import LinearSVC, SVC

# import ssl
# import certifi
# ssl._create_default_https_context = ssl.create_default_context
# ssl._create_default_https

# iris = datasets.load_iris()  # load data_set
# X, y = iris.data[:, :2], iris.targetmg
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33)
# scaler = preprocessing.StandardScaler().fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)
# knn = neighbors.KNeighborsClassifier(n_neighbors=5)
# knn.fit(X_train, y_train)
# y_pred = knn.predict(X_test)
# accuracy_score(y_test, y_pred)
#
# print(accuracy_score(y_test, y_pred))


def bai_1():
    print("======= Bài 1 ==========")
    iris = load_iris()
    iris_data = iris.data
    print("Dữ liệu 5 dòng đầu: ")
    print(iris_data[:5])


def bai_2():
    print("======= Bài 2 ==========")
    iris = load_iris()
    iris_target = iris.target
    print("Target 5 giá trị đầu: ", iris_target[:5])
    species_names = iris.target_names
    print("Tổng các loài: ", species_names)


def bai_3():
    print("======= Bài 3 ==========")
    iris = load_iris()
    sepal_length = iris.data[:, 0]
    sepal_width = iris.data[:, 1]
    plt.scatter(
        sepal_length[iris.target == 0],
        sepal_width[iris.target == 0],
        label="Setosa",
        c="red",
    )
    plt.scatter(
        sepal_length[iris.target == 1],
        sepal_width[iris.target == 1],
        label="Versicolor",
        c="blue",
    )
    plt.scatter(
        sepal_length[iris.target == 2],
        sepal_width[iris.target == 2],
        label="Virginica",
        c="green",
    )
    plt.xlabel("Chiều dài đài hoa (cm)")
    plt.ylabel("Chiều rộng đài hoa (cm)")
    plt.title("Biểu đồ phân tán của các loài Iris")
    plt.legend()
    plt.show()


def bai_4():
    print("======= Bài 4 ==========")
    iris = load_iris()
    pca = PCA(n_components=3)
    iris_pca = pca.fit_transform(iris.data)
    print(iris_pca[:5])


def bai_5():
    print("======= Bài 5 ==========")
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, train_size=140, test_size=10, random_state=42
    )
    print("Kích thước tập huấn luyện: ", X_train.shape)
    print("Kích thước tập kiểm tra: ", X_test.shape)


def bai_6():
    print("======= Bài 6 ==========")
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, train_size=140, test_size=10, random_state=42
    )
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print("Dự đoán: ", y_pred)


def bai_7():
    print("======= Bài 7 ==========")
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, train_size=140, test_size=10, random_state=42
    )
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print("Nhãn thực tế: ", y_test)
    print("Nhãn dự đoán: ", y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print("Độ chính xác: ", accuracy)


def bai_8():
    print("======= Bài 8 ==========")
    iris = load_iris()
    X = iris.data[:, [0, 1]]
    y = iris.target
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X, y)
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k")
    plt.xlabel("Chiều dài đài hoa (cm)")
    plt.ylabel("Chiều rộng đài hoa (cm)")
    plt.title("Ranh giới quyết định với KNN")
    plt.show()


def bai_9():
    print("======= Bài 9 ==========")
    diabetes = load_diabetes()
    print("Dữ liệu 5 hàng đầu: ")
    print(diabetes.data[:, 5])

diabetes = load_diabetes()
X_train = diabetes.data[:422]
X_test = diabetes.data[422:]
y_train = diabetes.target[:422]
y_test = diabetes.target[422:]

def bai_10():
    print("======= Bài 10 ==========")
    print("Kích thước tập huấn luyện: ", X_train.shape)
    print("Kích thước tập kiểm tra: ", X_test.shape)


def bai_11():
    print("=======Bài 11==========")
    global lr
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    print("Đã huấn luyện lr!!!")


def bai_12():
    print("=======Bài 12==========")
    coefficients = lr.coef_
    print("10 hệ số b: ", coefficients)


def bai_13():
    print("=======Bài 13==========")
    global y_pred
    y_pred = lr.predict(X_test)
    print("Giá trị thực tế: ", y_test)
    print("Giá trị dự đoán: ", y_pred)


def bai_14():
    print("=======Bài 14==========")
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print("Hệ số R^2: ", r2)
    print("Lỗi trung bình bình phương (MSE): ", mse)


def bai_15():
    print("=======Bài 15==========")
    X_train_age = X_train[:, [0]]
    X_test_age = X_test[:, [0]]
    lr_age = LinearRegression()
    lr_age.fit(X_train_age, y_train)
    y_pred_age = lr_age.predict(X_test_age)
    print("Dự đoán với tuổi:", y_pred_age)


def bai_16():
    print("=======Bài 16==========")
    for i in range(10):
        X_train_single = X_train[:, [i]]
        X_test_single = X_test[:, [i]]
        lr_single = LinearRegression()
        lr_single.fit(X_train_single, y_train)
        y_pred_single = lr_single.predict(X_test_single)
        plt.figure()
        plt.scatter(X_test_single, y_test, color="blue", label="Thực tế")
        plt.plot(X_test_single, y_pred_single, color="red", label="Dự đoán")
        plt.xlabel(f"Đặc trưng {i + 1}")
        plt.ylabel("Tiến trình bệnh")
        plt.title(f"Hồi quy tuyến tính với đặc trưng {i + 1}")
        plt.legend()
        plt.show()


def bai_17():
    print("=======Bài 17==========")
    global breast_cancer
    breast_cancer = load_breast_cancer()
    print("Các khoá của từ điển: ", breast_cancer.keys())


def bai_18():
    print("=======Bài 18==========")
    print("Kích thước dữ liệu: ", breast_cancer.data.shape)
    target_series = pd.Series(breast_cancer.target)
    benign_count = target_series.value_counts()[1]
    malignant_count = target_series.value_counts()[0]
    print("Số lượng khối u lành tính (benign):", benign_count)
    print("Số lượng khối u ác tính (malignant):", malignant_count)


def bai_19():
    print("=======Bài 19==========")
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(
        breast_cancer.data, breast_cancer.target, test_size=0.2, random_state=42
    )
    train_scores = []
    test_scores = []
    for k in range(1, 11):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        train_scores.append(knn.score(X_train, y_train))
        test_scores.append(knn.score(X_test, y_test))
    plt.plot(range(1, 11), train_scores, label="Độ chính xác tập huấn luyện")
    plt.plot(range(1, 11), test_scores, label="Độ chính xác tập kiểm tra")
    plt.xlabel("Số láng giềng (K)")
    plt.ylabel("Độ chính xác")
    plt.title("Hiệu suất KNN với số láng giềng từ 1 đến 10")
    plt.legend()
    plt.show()


def bai_20():
    print("=======Bài 20==========")
    X, y = mglearn.datasets.make_forge()
    logreg = LogisticRegression().fit(X, y)
    print("Độ chính xác LogisticRegression: ", logreg.score(X, y))
    svc = LinearSVC().fit(X, y)
    print("Độ chính xác LinearSVC: ", svc.score(X, y))


def bai_21():
    print("=======Bài 21==========")
    global faces
    faces = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    print("Mô tả bộ dữ liệu: ", faces.DESCR)


def bai_22():
    print("=======Bài 22==========")
    print("Kích thước images: ", faces.images.shape)
    print("Kích thước data: ", faces.data.shape)
    print("Kích thước target: ", faces.target.shape)
    print("Tên nhãn: ", faces.target_names)


def plot_faces(images, n_row=2, n_col=5):
    plt.figure(figsize=(2 * n_col, 2.5 * n_row))
    for j in range(n_row * n_col):
        plt.subplot(n_row, n_col, j + 1)
        plt.imshow(images[j], cmap="gray")
        plt.axis("off")
    plt.show()


def bai_23():
    print("=======Bài 23==========")
    plot_faces(faces.images)


def bai_24():
    print("=======Bài 24==========")
    global svc
    svc = SVC(kernel="linear")
    print("Khởi tạo SVC!!!")


def bai_25():
    print("=======Bài 25==========")
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(
        faces.data, faces.target, test_size=0.25, random_state=42
    )
    print("Kích thước tập huấn luyện: ", X_train.shape)
    print("Kích thước tập kiểm tra: ", X_test.shape)


def evaluate_cross_validation(model, X, y, k=5):
    scores = cross_val_score(model, X, y, cv=k)
    print(
        f"Độ chính xác K-fold (k={k}): {scores.mean():.2f} (+/- {scores.std() * 2:.2f})"
    )


def bai_26():
    print("=======Bài 26==========")


def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print("Độ chính xác tập huấn luyện: ", train_score)
    print("Độ chính xác tập kiểm tra: ", test_score)


def bai_27():
    print("=======Bài 27==========")


def bai_28():
    print("=======Bài 28==========")
    evaluate_cross_validation(svc, faces.data, faces.target)
    train_and_evaluate(svc, X_train, X_test, y_train, y_test)


def create_glasses_target(target):
    np.random.seed(42)
    return np.random.randint(0, 2, size=len(target))


def bai_29():
    print("=======Bài 29==========")

    global faces_glasses_target
    faces_glasses_target = create_glasses_target(faces.target)
    print("Mảng mục tiêu mới (10 giá trị đầu tiên): ", faces_glasses_target[:10])


def bai_30():
    print("=======Bài 30==========")
    global svc_2, X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = train_test_split(
        faces.data, faces_glasses_target, test_size=0.25, random_state=42
    )
    svc_2 = SVC(kernel="linear")
    svc_2.fit(X_train, y_train)
    print("Huấn luyện SVC2")


def bai_31():
    print("=======Bài 31==========")
    evaluate_cross_validation(svc_2, X_train, y_train, 5)


def bai_32():
    print("=======Bài 32==========")
    global svc_3, X_eval, y_eval
    X_eval = faces.data[30:40]
    y_eval = faces_glasses_target[30:40]

    X_train_remaining = np.concatenate((faces.data[:30], faces.data[40:]))
    y_train_remaining = np.concatenate(
        (faces_glasses_target[:30], faces_glasses_target[40:])
    )

    svc_3 = SVC(kernel="linear")
    svc_3.fit(X_train_remaining, y_train_remaining)
    accuracy = svc_3.score(X_eval, y_eval)
    print("Độ chính xác trên tập đánh giá 10 ảnh:", accuracy)


def plot_faces__(images, predictions, n_col=10):
    plt.figure(figsize=(2 * n_col, 2.5))
    for j in range(len(images)):
        plt.subplot(1, n_col, j + 1)
        plt.imshow(images[j], cmap="gray")
        plt.title(f"Pred: {predictions[j]}")
        plt.axis("off")
    plt.show()


def bai_33():
    print("=======Bài 33==========")
    y_pred = svc_3.predict(X_eval)
    eval_faces = [np.reshape(a, (50, 37)) for a in X_eval]

    plot_faces__(eval_faces, y_pred)

    for i in range(len(y_eval)):
        if y_eval[i] != y_pred[i]:
            print(
                f"Ảnh ở chỉ số {i+30} bị phân loại sai. Thực tế {y_eval[i]}, Dự đoán: {y_pred[i]}"
            )

# run
for i in range(1, 34):
    eval(f"bai_{i}()")

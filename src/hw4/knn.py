import csv
import math
from collections import Counter


# 讀取 CSV 檔案的函式
def load_csv(filename):
    data = []
    with open(filename, mode='r', encoding='utf-8-sig') as file:
        csv_reader = csv.reader(file)
        header = next(csv_reader)  # 讀取第一行為標題
        for row in csv_reader:
            # 確保每一行有正確的資料數量，排除空行或錯誤的資料行
            if len(row) == len(header):  # 確保每行資料的欄位數與標題數一致
                data.append(row)
            else:
                print(f"Skipping malformed row: {row}")  # 顯示錯誤的資料行
    return header, data


# 資料清理與轉換函式
def preprocess_data(data):
    processed_data = []
    for row in data:
        try:
            # 轉換類別資料為數值
            gender = 0 if row[0] == "Male" else 1
            senior_citizen = 1 if row[1] == "Yes" else 0
            partner = 1 if row[2] == "Yes" else 0
            dependents = 1 if row[3] == "Yes" else 0
            # tenure = int(row[4])
            phone_service = 1 if row[5] == "Yes" else 0
            multiple_lines = 1 if row[6] == "Yes" else 0
            internet_service = [1 if row[7] == "Fiber optic" else 0, 1 if row[7] == "DSL" else 0]
            online_security = 1 if row[8] == "Yes" else 0
            online_backup = 1 if row[9] == "Yes" else 0
            device_protection = 1 if row[10] == "Yes" else 0
            tech_support = 1 if row[11] == "Yes" else 0
            streaming_tv = 1 if row[12] == "Yes" else 0
            streaming_movies = 1 if row[13] == "Yes" else 0
            contract = [1 if row[14] == "One year" else 0, 1 if row[14] == "Two year" else 0]
            paperless_billing = 1 if row[15] == "Yes" else 0
            payment_method = [1 if row[16] == "Electronic check" else 0,
                              1 if row[16] == "Bank transfer (automatic)" else 0,
                              1 if row[16] == "Credit card (automatic)" else 0]
            monthly_charges = float(row[17])
            total_charges = float(row[18]) if row[18] != "" else 0.0
            
            # 組合所有特徵
            processed_data.append([
                gender, senior_citizen, partner, dependents,
                phone_service, multiple_lines, *internet_service,
                online_security, online_backup, device_protection,
                tech_support, streaming_tv,  streaming_movies,
                *contract, paperless_billing, *payment_method,
                monthly_charges, total_charges
            ])
        except ValueError as e:
            print(f"Error processing row: {row} - {e}")
    return processed_data

# 歐幾里得距離計算
def calculate_distance(sample1, sample2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(sample1, sample2)))

# KNN 演算法
def knn_predict(train_data, train_labels, test_data, k=5):
    """一般knn"""
    predictions = []
    for test_point in test_data:
        distances = []
        for i, train_point in enumerate(train_data):
            distance = calculate_distance(test_point, train_point)
            distances.append((distance, train_labels[i][0]))
        distances.sort(key=lambda x: x[0])  # 按距離排序
        k_nearest_neighbors = distances[:k]
        k_nearest_labels = [label for _, label in k_nearest_neighbors]
        most_common_label = Counter(k_nearest_labels).most_common(1)[0][0]
        predictions.append(most_common_label)
    return predictions

def knn_predict_weighted(train_data, train_labels, test_data, k=5):
    """權重knn"""
    predictions = []
    for test_point in test_data:
        distances = []
        for i, train_point in enumerate(train_data):
            distance = calculate_distance(test_point, train_point)
            distances.append((distance, train_labels[i][0]))
        distances.sort(key=lambda x: x[0])
        k_nearest_neighbors = distances[:k]
        weighted_vote = {}
        for dist, label in k_nearest_neighbors:
            weight = 1 / (dist + 1e-5)  # 避免除以 0
            weighted_vote[label] = weighted_vote.get(label, 0) + weight
        predictions.append(max(weighted_vote, key=weighted_vote.get))
    return predictions

def normalize_data(data):
    """
    將資料標準化（每個特徵列進行 Min-Max Scaling）。
    :param data: 需要標準化的資料（每一行為一個樣本，每一列為一個特徵）。
    :return: 標準化後的資料。
    """
    # 檢查資料是否為空
    if len(data) == 0 or not all(isinstance(row, list) for row in data):
        print("Error: Input data is empty or not in the expected format.")
        return data  # 若資料格式不正確，直接返回原資料

    # 標準化每一列（每個特徵）
    normalized_data = []
    for i in range(len(data[0])):  # 每個特徵逐列標準化
        col = [row[i] for row in data]
        
        # 檢查此列是否包含有效的數值
        if not all(isinstance(x, (int, float)) for x in col):
            print(f"Warning: Skipping non-numeric column {i}.")
            normalized_data.append(col)  # 如果列中有非數字，則不標準化，保持原始數據
            continue
        
        min_val, max_val = min(col), max(col)
        # 處理分母為零的情況
        normalized_data.append([(x - min_val) / (max_val - min_val) if max_val != min_val else 0 for x in col])
    
    # 將列轉置回原來的資料結構
    return list(zip(*normalized_data))



# 模型準確率計算
def evaluate_accuracy(predictions, ground_truth):
    correct = sum([1 for pred, actual in zip(predictions, ground_truth) if pred == actual[0]])
    return correct / len(ground_truth)

# 儲存結果為 CSV
def save_predictions(predictions, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Churn"])
        for pred in predictions:
            writer.writerow([pred])

# 主程式
if __name__ == "__main__":
    # 讀取資料
    train_header, train_data = load_csv('res/hw4/train.csv')
    train_gt_header, train_gt_data = load_csv('res/hw4/train_gt.csv')
    val_header, val_data = load_csv('res/hw4/val.csv')
    val_gt_header, val_gt_data = load_csv('res/hw4/val_gt.csv')
    test_header, test_data = load_csv('res/hw4/test.csv')

    # 資料前處理
    processed_train_data = preprocess_data(train_data)
    processed_val_data = preprocess_data(val_data)
    processed_test_data = preprocess_data(test_data)

    # def recursive_feature_elimination(train_data, train_labels, val_data, val_labels, k=5, min_features=5):
    #     current_features = list(range(len(train_header)))  # 初始特徵索引
    #     best_accuracy = 0
    #     best_features = current_features[:]
        
    #     while len(current_features) > min_features:
    #         accuracies = []
    #         print(f"Current features: {current_features}")  # 顯示當前特徵列表
            
    #         for feature in current_features:
    #             reduced_features = [f for f in current_features if f != feature]
    #             reduced_train_data = [[row[i] for i in reduced_features] for row in train_data]
    #             reduced_val_data = [[row[i] for i in reduced_features] for row in val_data]
    #             val_predictions = knn_predict(reduced_train_data, train_labels, reduced_val_data, k)
    #             accuracy = evaluate_accuracy(val_predictions, val_labels)
    #             accuracies.append((feature, accuracy))
            
    #         # 找到移除影響最小的特徵
    #         feature_to_remove, _ = min(accuracies, key=lambda x: x[1])
    #         current_features.remove(feature_to_remove)
    #         print(f"Removed feature {feature_to_remove}, Remaining features: {current_features}")
            
    #         # 更新最佳特徵組合
    #         if accuracies[-1][1] > best_accuracy:
    #             best_accuracy = accuracies[-1][1]
    #             best_features = current_features[:]
        
    #     return best_features


    # # 執行特徵消除
    # best_features = recursive_feature_elimination(processed_train_data, train_gt_data, processed_val_data, val_gt_data)
    # print("Best features:", best_features)

    # 標準化訓練與驗證資料
    processed_train_data = normalize_data(processed_train_data)
    processed_val_data = normalize_data(processed_val_data)
    processed_test_data = normalize_data(processed_test_data)

    # K 值設置與驗證
    k = 30  # 可根據驗證集調整最佳 K 值
    val_predictions = knn_predict(processed_train_data, train_gt_data, processed_val_data, k)
    save_predictions(val_predictions, filename="val_pred.csv")
    print("Validation predictions have been saved to val_pred.csv.")

    # 使用 KNN 進行測試集預測並輸出 test_pred.csv
    test_predictions = knn_predict_weighted(processed_train_data, train_gt_data, processed_test_data, k)
    save_predictions(test_predictions, filename="test_pred.csv")
    print("Test predictions have been saved to test_pred.csv.")

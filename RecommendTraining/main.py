import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

DATA_URL = 'input/employee_course_ratings.csv'


def load_data():
    data = pd.read_csv(DATA_URL)
    print(data.head())
    return data


def preprocess(df):
    emp_list = df.groupby(['EmployeeID', 'EmpName']).size().reset_index()
    print("Total Employees: ", len(emp_list))
    course_list = df.groupby(['CourseID', 'CourseName']).size().reset_index()
    print("Total Courses: ", len(course_list))
    return emp_list, course_list


def model(emp_list, course_list):
    # build employee embedding vector
    emp_input = Input(shape=[1], name="Emp-Input")
    emp_embed = Embedding(2001,  # max value of employee ID
                          5,
                          name="Emp-Embedding")(emp_input)
    emp_vec = Flatten(name="Emp-Flatten")(emp_embed)

    # build course embedding vector
    course_input = Input(shape=[1], name="Course-Input")
    course_embed = Embedding(len(course_list) + 1,
                             5,
                             name="Course-Embedding")(course_input)
    course_vec = Flatten(name="Course-Flatten")(course_embed)

    # merge the vectors
    merged_vec = Concatenate()([emp_vec, course_vec])

    fc_layer1 = Dense(128, activation="relu")(merged_vec)
    fc_layer2 = Dense(32, activation="relu")(fc_layer1)
    output = Dense(1)(fc_layer2)
    mdl = Model([emp_input, course_input], output)
    mdl.compile(optimizer="adam", loss="mean_squared_error")
    print(mdl.summary())
    return mdl


def main():
    ratings_data = load_data()
    emp_list, course_list = preprocess(ratings_data)

    ratings_train, ratings_test = train_test_split(ratings_data, test_size=0.1)

    m = model(emp_list, course_list)
    m.fit(
        x=[ratings_train.EmployeeID, ratings_train.CourseID],
        y=ratings_train.Rating,
        epochs=25,
        verbose=1,
        validation_split=0.1
    )

    m.evaluate(
        x=[ratings_test.EmployeeID, ratings_test.CourseID],
        y=ratings_test.Rating
    )

    # predicting rating for given employee 1029 and course 8

    m.predict([pd.Series([1029]), pd.Series([8])])

    # predict and recommend courses for employee
    emp_to_predict = "Harriot Laflin"
    # get employee ID
    pred_emp_id = emp_list[emp_list['EmpName'] == emp_to_predict]["EmployeeID"].iloc[0]
    # find courses already taken by employee - we don't want to predict these
    completed_courses = ratings_data[ratings_data["EmployeeID"] == pred_emp_id]["CourseID"].unique()
    # courses not taken by employee
    new_courses = course_list.query("CourseID not in @completed_courses")["CourseID"]
    # Create a list with the same employee ID repeated for the same number of times as the number of new courses.
    # This provides the employee and course Series with the same size.
    emp_dummy_list = pd.Series(np.array([pred_emp_id for i in range(len(new_courses))]))
    # predict ratings for the new courses for this employee
    pred_ratings = m.predict([emp_dummy_list, new_courses])
    flat_ratings = np.array([x[0] for x in pred_ratings])

    print("Course Ratings: ", flat_ratings)
    # recommend to 5 courses
    print("Recommended:")
    for idx in (-flat_ratings).argsort()[:5]:
        course_id = new_courses.iloc[idx]
        course_name = course_list.query("CourseID == @course_id")["CourseName"].iloc[0]
        print(" ", round(flat_ratings[idx], 1), "  ", course_id, "  ", course_name)


if __name__ == "__main__":
    main()

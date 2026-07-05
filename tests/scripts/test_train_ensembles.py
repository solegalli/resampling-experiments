import time
import joblib

# Create a mock search object
class MockSearch:
    def __init__(self):
        self.best_params_ = {"a": 1}


def test_fit_time_added_to_saved_model(tmp_path):
    # This simulates what we do in the script
    search = MockSearch()
    output_file = tmp_path / "test_model.pkl"

    start_time = time.time()
    # Simulate training
    time.sleep(0.01)
    search.fit_time = time.time() - start_time

    # Save the object
    joblib.dump(search, output_file)

    # Load it back
    loaded_search = joblib.load(output_file)

    # Assert
    assert hasattr(loaded_search, "fit_time")
    assert isinstance(loaded_search.fit_time, float)
    assert loaded_search.fit_time > 0

from flask import Flask, request, jsonify
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity
from functions import ask_question, main
from werkzeug.security import generate_password_hash, check_password_hash

# Initialize Flask app
app = Flask(__name__)

# Setup Flask-JWT-Extended
app.config['JWT_SECRET_KEY'] = 'faizan'  # secret key
jwt = JWTManager(app)

# In a real application, these users would be stored in a database
# For demonstration purposes, we'll use a simple dictionary
users = {
    'user1@example.com': generate_password_hash('password1'),
    'user2@example.com': generate_password_hash('password2')
}

@app.route('/register', methods=['POST'])
def register():
    """
    Handle POST requests to the '/register' endpoint.
    
    Registers a new user with the provided email and password.
    
    Returns:
        JSON: A JSON response indicating success or failure.
    """
    data = request.json
    email = data.get('email')
    password = data.get('password')
    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400
    if email in users:
        return jsonify({"error": "User already exists"}), 400
    users[email] = generate_password_hash(password)
    return jsonify({"message": "User registered successfully"}), 201

@app.route('/login', methods=['POST'])
def login():
    """
    Handle POST requests to the '/login' endpoint.
    
    Authenticates the user with the provided email and password.
    If authentication is successful, returns a JWT token.
    
    Returns:
        JSON: A JSON response containing the JWT token.
    """
    data = request.json
    email = data.get('email')
    password = data.get('password')
    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400
    if email not in users or not check_password_hash(users[email], password):
        return jsonify({"error": "Invalid email or password"}), 401
    access_token = create_access_token(identity=email)
    return jsonify(access_token=access_token), 200

@app.route('/query', methods=['GET'])
@jwt_required()
def query():
    """
    Handle GET requests to the '/query' endpoint.
    
    Retrieves the query from the URL parameters, processes it, and returns the response.
    
    Returns:
        JSON: A JSON response containing the answer to the query.
    """
    # Retrieve query text from the URL parameters
    query_text = request.args.get('q', None)

    # Check if a query is provided
    if query_text is None:
        return jsonify({'error': 'No query provided'}), 400

    # Process the query and retrieve the response
    crc = main()  # Assuming main() returns some data needed for processing queries
    result = ask_question(query_text, crc)
    text = result['answer']
    
    # Return the response as JSON
    return jsonify({'response': text})

if __name__ == "__main__":
    # Run the Flask app
    app.run(debug=False)

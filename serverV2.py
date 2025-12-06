#!/usr/bin/env python3
"""
Python WebSocket Server for Unity Neural Network
Receives input data from Unity and returns neural network predictions
Run this server before starting Unity: python python_nn_server.py
"""

import asyncio
import websockets
import json
import numpy as np
from typing import Dict, List, Optional
import uuid

class NeuralNetwork:
    """Simple neural network implementation in Python"""
    
    def __init__(self, layers, network_id=None):
        self.id = network_id or str(uuid.uuid4())
        self.layers = layers
        self.weights = []
        self.biases = []
        self.fitness = 0.0
        
        # Initialize weights and biases
        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i], layers[i + 1]) * 0.5
            b = np.zeros((1, layers[i + 1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def tanh(self, x):
        """Tanh activation function"""
        return np.tanh(x)
    
    def forward(self, inputs):
        """Forward propagation"""
        activation = np.array(inputs).reshape(1, -1)
        
        for i in range(len(self.weights)):
            z = np.dot(activation, self.weights[i]) + self.biases[i]
            
            # Use ReLU for hidden layers, tanh for output
            if i < len(self.weights) - 1:
                activation = self.relu(z)
            else:
                activation = self.tanh(z)
        
        return activation.flatten().tolist()
    
    def mutate(self, mutation_rate=0.1, mutation_strength=0.2):
        """
        Simple mutation of network weights
        
        Args:
            mutation_rate: Probability of mutating each weight
            mutation_strength: Magnitude of mutations
        """
        for i in range(len(self.weights)):
            # Weight mutations
            mask = np.random.random(self.weights[i].shape) < mutation_rate
            mutations = np.random.randn(*self.weights[i].shape) * mutation_strength
            self.weights[i] += mask * mutations
            
            # Bias mutations
            mask = np.random.random(self.biases[i].shape) < mutation_rate
            mutations = np.random.randn(*self.biases[i].shape) * mutation_strength
            self.biases[i] += mask * mutations
    
    def crossover_mutate(self, mama, papa):
        """
        Crossover mutation matching Unity's NeuralNetwork.Mutate(mama, papa)
        This creates a child by combining and mutating parent weights
        
        Args:
            mama: First parent NeuralNetwork
            papa: Second parent NeuralNetwork
        """
        for i in range(len(self.weights)):
            for j in range(self.weights[i].shape[0]):
                for k in range(self.weights[i].shape[1]):
                    weight = self.weights[i][j, k]
                    weight_mama = mama.weights[i][j, k]
                    weight_papa = papa.weights[i][j, k]
                    
                    # Mutate weight value (matching Unity logic)
                    random_number = np.random.uniform(0, 100)
                    
                    if random_number <= 5:
                        # Flip sign of weight
                        weight *= -1
                    elif random_number <= 10:
                        # Pick random weight between -0.5 and 0.5
                        weight = np.random.uniform(-0.5, 0.5)
                    elif random_number <= 15:
                        # Randomly increase by 0% to 100%
                        factor = np.random.uniform(0, 1) + 1
                        weight *= factor
                    elif random_number <= 20:
                        # Randomly decrease by 0% to 100%
                        factor = np.random.uniform(0, 1)
                        weight *= factor
                    elif random_number <= 80:
                        # Take weight from mama
                        weight = weight_mama
                    else:
                        # Take weight from papa
                        weight = weight_papa
                    
                    self.weights[i][j, k] = weight
            
            # Also mutate biases similarly
            for j in range(self.biases[i].shape[1]):
                bias = self.biases[i][0, j]
                bias_mama = mama.biases[i][0, j] if i < len(mama.biases) else bias
                bias_papa = papa.biases[i][0, j] if i < len(papa.biases) else bias
                
                random_number = np.random.uniform(0, 100)
                
                if random_number <= 5:
                    bias *= -1
                elif random_number <= 10:
                    bias = np.random.uniform(-0.5, 0.5)
                elif random_number <= 15:
                    factor = np.random.uniform(0, 1) + 1
                    bias *= factor
                elif random_number <= 20:
                    factor = np.random.uniform(0, 1)
                    bias *= factor
                elif random_number <= 80:
                    bias = bias_mama
                else:
                    bias = bias_papa
                
                self.biases[i][0, j] = bias
    
    def crossover(self, other, crossover_rate=0.5):
        """
        Create child network from two parents (alternative approach)
        
        Args:
            other: Another NeuralNetwork instance
            crossover_rate: Probability of taking weight from this parent vs other
        """
        child = NeuralNetwork(self.layers)
        
        for i in range(len(self.weights)):
            # Crossover weights
            mask = np.random.random(self.weights[i].shape) < crossover_rate
            child.weights[i] = np.where(mask, self.weights[i], other.weights[i])
            
            # Crossover biases
            mask = np.random.random(self.biases[i].shape) < crossover_rate
            child.biases[i] = np.where(mask, self.biases[i], other.biases[i])
        
        return child
    
    def get_weights_flat(self):
        """Serialize weights to flat array for Unity"""
        flat = []
        for w, b in zip(self.weights, self.biases):
            flat.extend(w.flatten().tolist())
            flat.extend(b.flatten().tolist())
        return flat
    
    def set_weights_flat(self, flat_weights):
        """Deserialize weights from flat array"""
        idx = 0
        for i in range(len(self.weights)):
            w_size = self.weights[i].size
            b_size = self.biases[i].size
            
            self.weights[i] = np.array(flat_weights[idx:idx+w_size]).reshape(self.weights[i].shape)
            idx += w_size
            
            self.biases[i] = np.array(flat_weights[idx:idx+b_size]).reshape(self.biases[i].shape)
            idx += b_size


# Global neural network instances
networks: Dict[str, NeuralNetwork] = {}


async def handle_client(websocket):
    """Handle WebSocket connection from Unity client"""
    global networks
    
    print(f"Client connected from {websocket.remote_address}")
    
    try:
        async for message in websocket:
            try:
                # Parse JSON request from Unity
                data = json.loads(message)
                action = data.get('action', 'forward')
                
                response = {}
                
                # Handle different actions
                if action == 'create':
                    # Create new network
                    layers = data.get('layers', [4, 4, 1])
                    net = NeuralNetwork(layers)
                    networks[net.id] = net
                    print(f"Created network {net.id} with layers: {layers}")
                    
                    response = {
                        'status': 'success',
                        'network_id': net.id
                    }
                
                elif action == 'forward':
                    # Forward pass
                    network_id = data.get('network_id')
                    inputs = data.get('inputs', [])
                    
                    # Validate inputs
                    if not inputs or len(inputs) == 0:
                        response = {
                            'status': 'error',
                            'message': f'Empty inputs array. Expected {layers[0] if "layers" in data else "non-zero"} inputs.',
                            'output': [0.0]  # Return zero as fallback
                        }
                        print(f"Warning: Received empty inputs array")
                        await websocket.send(json.dumps(response))
                        continue
                    
                    # Fallback to old behavior if no network_id
                    if network_id and network_id in networks:
                        net = networks[network_id]
                        
                        # Validate input size matches network
                        if len(inputs) != net.layers[0]:
                            response = {
                                'status': 'error',
                                'message': f'Input size mismatch. Expected {net.layers[0]}, got {len(inputs)}',
                                'output': [0.0]
                            }
                            print(f"Error: Input size {len(inputs)} doesn't match network input layer {net.layers[0]}")
                            await websocket.send(json.dumps(response))
                            continue
                    else:
                        # Legacy support - use global network
                        layers = data.get('layers', [4, 4, 1])
                        if 'global' not in networks:
                            networks['global'] = NeuralNetwork(layers)
                        net = networks['global']
                        
                        # Validate input size
                        if len(inputs) != layers[0]:
                            response = {
                                'status': 'error',
                                'message': f'Input size mismatch. Expected {layers[0]}, got {len(inputs)}',
                                'output': [0.0]
                            }
                            print(f"Error: Input size {len(inputs)} doesn't match expected {layers[0]}")
                            await websocket.send(json.dumps(response))
                            continue
                    
                    try:
                        output = net.forward(inputs)
                        response = {
                            'status': 'success',
                            'output': output
                        }
                    except Exception as forward_error:
                        response = {
                            'status': 'error',
                            'message': f'Forward pass failed: {str(forward_error)}',
                            'output': [0.0]
                        }
                        print(f"Error in forward pass: {forward_error}")
                
                elif action == 'mutate':
                    # Simple mutation
                    network_id = data.get('network_id')
                    mutation_rate = data.get('mutation_rate', 0.1)
                    mutation_strength = data.get('mutation_strength', 0.2)
                    
                    if network_id in networks:
                        networks[network_id].mutate(mutation_rate, mutation_strength)
                        response = {
                            'status': 'success',
                            'message': 'Network mutated'
                        }
                    else:
                        response = {
                            'status': 'error',
                            'message': f'Network {network_id} not found'
                        }
                
                elif action == 'crossover_mutate':
                    # Crossover mutation (mama + papa)
                    network_id = data.get('network_id')
                    mama_id = data.get('mama_id')
                    papa_id = data.get('papa_id')
                    
                    if network_id in networks and mama_id in networks and papa_id in networks:
                        child = networks[network_id]
                        mama = networks[mama_id]
                        papa = networks[papa_id]
                        
                        child.crossover_mutate(mama, papa)
                        
                        response = {
                            'status': 'success',
                            'message': 'Crossover mutation completed'
                        }
                        print(f"Crossover mutation: {network_id} from parents {mama_id} + {papa_id}")
                    else:
                        response = {
                            'status': 'error',
                            'message': 'One or more networks not found'
                        }
                
                elif action == 'crossover':
                    # Create child from two parents
                    parent1_id = data.get('parent1_id')
                    parent2_id = data.get('parent2_id')
                    crossover_rate = data.get('crossover_rate', 0.5)
                    
                    if parent1_id in networks and parent2_id in networks:
                        parent1 = networks[parent1_id]
                        parent2 = networks[parent2_id]
                        
                        child = parent1.crossover(parent2, crossover_rate)
                        networks[child.id] = child
                        
                        response = {
                            'status': 'success',
                            'child_id': child.id
                        }
                        print(f"Created child {child.id} from parents {parent1_id} + {parent2_id}")
                    else:
                        response = {
                            'status': 'error',
                            'message': 'Parents not found'
                        }
                
                elif action == 'update_fitness':
                    # Update fitness
                    network_id = data.get('network_id')
                    fitness = data.get('fitness', 0.0)
                    
                    if network_id in networks:
                        networks[network_id].fitness = fitness
                        response = {
                            'status': 'success'
                        }
                    else:
                        response = {
                            'status': 'error',
                            'message': f'Network {network_id} not found'
                        }
                
                elif action == 'get_weights':
                    # Get weights for Unity
                    network_id = data.get('network_id')
                    
                    if network_id in networks:
                        response = {
                            'status': 'success',
                            'weights': networks[network_id].get_weights_flat()
                        }
                    else:
                        response = {
                            'status': 'error',
                            'message': f'Network {network_id} not found'
                        }
                
                elif action == 'set_weights':
                    # Set weights from Unity
                    network_id = data.get('network_id')
                    weights = data.get('weights', [])
                    
                    if network_id in networks:
                        networks[network_id].set_weights_flat(weights)
                        response = {
                            'status': 'success'
                        }
                    else:
                        response = {
                            'status': 'error',
                            'message': f'Network {network_id} not found'
                        }
                
                elif action == 'get_best_networks':
                    # Get top N networks by fitness
                    count = data.get('count', 10)
                    sorted_nets = sorted(networks.values(), key=lambda n: n.fitness, reverse=True)
                    best_ids = [net.id for net in sorted_nets[:count]]
                    
                    response = {
                        'status': 'success',
                        'network_ids': best_ids,
                        'fitnesses': [net.fitness for net in sorted_nets[:count]]
                    }
                
                else:
                    response = {
                        'status': 'error',
                        'message': f'Unknown action: {action}'
                    }
                
                await websocket.send(json.dumps(response))
                
            except json.JSONDecodeError as e:
                error_response = {
                    'status': 'error',
                    'message': f'Invalid JSON: {str(e)}'
                }
                await websocket.send(json.dumps(error_response))
                print(f"JSON decode error: {e}")
                
            except Exception as e:
                error_response = {
                    'status': 'error',
                    'message': str(e)
                }
                await websocket.send(json.dumps(error_response))
                print(f"Error processing request: {e}")
    
    except websockets.exceptions.ConnectionClosed:
        print(f"Client disconnected from {websocket.remote_address}")
    except Exception as e:
        print(f"Unexpected error: {e}")


async def main():
    """Start WebSocket server"""
    print("Starting Python Neural Network WebSocket Server...")
    print("Listening on ws://localhost:8765")
    print("Supported actions:")
    print("  - create, forward, mutate, crossover_mutate, crossover")
    print("  - update_fitness, get_weights, set_weights, get_best_networks")
    print("Press Ctrl+C to stop the server\n")
    
    async with websockets.serve(handle_client, "localhost", 8765):
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped by user")

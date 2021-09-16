#include <queue>
#include <limits>
#include <cmath>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <iostream>


const float INF = std::numeric_limits<float>::infinity();

// represents a single pixel
class Node {
  public:
    int idx; // index in the flattened grid
    int max_g_idx; // cost of traversing this pixel
    float g; // Cost from start to pixel
    float h; // heuristic cost from pixel to goal
    float cost; // cost of traversing this pixel
    int path_length; // the length of the path to reach this node

    Node(int i, int g_idx, float g, float h, int path_length): idx(i), max_g_idx(g_idx), g(g), h(h), cost(g + h), path_length(path_length) {}
};

// the top of the priority queue is the greatest element by default,
// but we want the smallest, so flip the sign
bool operator<(const Node &n1, const Node &n2) {
  return n1.cost > n2.cost;
}

// See for various grid heuristics:
// http://theory.stanford.edu/~amitp/GameProgramming/Heuristics.html#S7
// L_\inf norm (diagonal distance)
inline float linf_norm(int i0, int j0, int i1, int j1) {
  return std::max(std::abs(i0 - i1), std::abs(j0 - j1));
}

// L_1 norm (manhattan distance)
inline float l1_norm(int i0, int j0, int i1, int j1) {
  return std::abs(i0 - i1) + std::abs(j0 - j1);
}


// If the path is incomplete, 
static Node *complete_path(const float* weights, Node *closestNode, int w, int h, int goal) {
  int* nbrs = new int[4];
  Node *cur = closestNode;
  while (cur->idx != goal) {
    int row = cur->idx / w;
    int col = cur->idx % w;

    int grow = goal / w;
    int gcol = goal % w;
    // check bounds and find up to eight neighbors: top to bottom, left to right
    nbrs[0] = (row > 0 && grow < row)                  ? cur->idx - w       : -1;
    nbrs[1] = (col > 0 && grow < col)                  ? cur->idx - 1       : -1;
    nbrs[2] = (col + 1 < w && gcol > col)              ? cur->idx + 1       : -1;
    nbrs[3] = (row + 1 < h && grow > row)              ? cur->idx + w       : -1;

     int i = 0;
     for (int i = 0; i < 4; i++) {
        if (nbrs[i] >= 0) {
          // The maximum value on the path
          // std::cout << "Before " << cur->g << " " << cur->max_g_idx << std::endl;
          if (weights[nbrs[i]] > cur->g) {
            cur->max_g_idx = nbrs[i];
            cur->g = std::max(cur->g, weights[nbrs[i]]);
          }
          cur->idx = nbrs[i];
          // std::cout <<"After " << cur->g << " " << cur->max_g_idx << std::endl;
          break;
        }
      }

  }
  return cur;
}


// weights:        flattened h x w grid of costs
// h, w:           height and width of grid
// start, goal:    index of start/goal in flattened grid
// diag_ok:        if true, allows diagonal moves (8-conn.)
// returns cell in the discovered path with the highest cost
static int astar(float* weights, float dist_weight, int h, int w,
                      int start, int goal, int diag_ok) {
  int* paths = new int[h * w];
  int path_length = -1;

  Node start_node(start, start, weights[start], 0., 1);

  float* costs = new float[h * w];
  for (int i = 0; i < h * w; ++i)
    costs[i] = INF;
  costs[start] = weights[start];

  std::priority_queue<Node> nodes_to_visit;
  nodes_to_visit.push(start_node);

  int* nbrs = new int[8];

  // Normalize the distance metric
  float dist_norm = h + w;
  Node* closest = NULL;
  while (!nodes_to_visit.empty()) {
    // .top() doesn't actually remove the node
    Node cur = nodes_to_visit.top();
    nodes_to_visit.pop();


    // Keep track of the closest node to the goal
    if (closest == NULL || closest->h > cur.h) {
      closest = &cur;
    }

    if (cur.idx == goal) {
      path_length = cur.path_length;
      break;
    }

    int row = cur.idx / w;
    int col = cur.idx % w;
    // check bounds and find up to eight neighbors: top to bottom, left to right
    nbrs[0] = (diag_ok && row > 0 && col > 0)          ? cur.idx - w - 1   : -1;
    nbrs[1] = (row > 0)                                ? cur.idx - w       : -1;
    nbrs[2] = (diag_ok && row > 0 && col + 1 < w)      ? cur.idx - w + 1   : -1;
    nbrs[3] = (col > 0)                                ? cur.idx - 1       : -1;
    nbrs[4] = (col + 1 < w)                            ? cur.idx + 1       : -1;
    nbrs[5] = (diag_ok && row + 1 < h && col > 0)      ? cur.idx + w - 1   : -1;
    nbrs[6] = (row + 1 < h)                            ? cur.idx + w       : -1;
    nbrs[7] = (diag_ok && row + 1 < h && col + 1 < w ) ? cur.idx + w + 1   : -1;

    float heuristic_cost;
    for (int i = 0; i < 8; ++i) {
      if (nbrs[i] >= 0) {
        // The maximum value on the path
        float new_cost = std::max(cur.g, weights[nbrs[i]]);
        int max_g_idx = cur.max_g_idx;
        if (weights[nbrs[i]] > cur.g) {
          max_g_idx = nbrs[i];
        }
        if (new_cost < costs[nbrs[i]]) {
          // estimate the cost to the goal based on legal moves
          if (diag_ok) {
            heuristic_cost = dist_weight * linf_norm(nbrs[i] / w, nbrs[i] % w,
                                       goal    / w, goal    % w) / dist_norm;
          }
          else {
            heuristic_cost = dist_weight * l1_norm(nbrs[i] / w, nbrs[i] % w,
                                     goal    / w, goal    % w) / dist_norm;
          }

          // paths with lower expected cost are explored first
          nodes_to_visit.push(Node(nbrs[i], max_g_idx, new_cost, heuristic_cost, cur.path_length + 1));
          
          costs[nbrs[i]] = new_cost;
          paths[nbrs[i]] = cur.idx;
        }
      }
    }
  }
  
  // Instead of returning the path, return the cell in 
  // the discovered path with the highest cost
  if (path_length < 0) {
    closest = complete_path(weights, closest, w, h, goal);
  }

  delete[] costs;
  delete[] nbrs;
  delete[] paths;

  return closest->max_g_idx;
}

static PyObject *astar_trials(PyObject *self, PyObject *args) {
  // Takes in a batch of masks and a batch of sampled start and end
  // points. Finds the optimal path between each start and end
  const PyArrayObject* weights_object;
  const PyArrayObject* dist_weights_object;
  int h;
  int w;
  int start;
  int goal;
  int diag_ok;

  if (!PyArg_ParseTuple(
        args, "OOiiiii", // i = int, O = object
        &weights_object, &dist_weights_object,
        &h, &w,
        &start, &goal,
        &diag_ok))
    return NULL;
  
  float* weights = (float*) weights_object->data;
  float* dist_weights = (float*) dist_weights_object->data;
  
  std::vector<int> g_idxs;

  // Assume there are three weights to try
  for (int i = 0; i < 3; i++) {
    g_idxs.push_back(astar(weights, *(dist_weights + i), h, w, start, goal, diag_ok));
  }
  
  PyObject *return_val;
  npy_intp dims[2] = {g_idxs.size(), 2};
  PyArrayObject* pos = (PyArrayObject*) PyArray_SimpleNew(2, dims, NPY_INT32);
  npy_int32 *iptr, *jptr;
  for (npy_intp i = dims[0] - 1; i >= 0; --i) {
      iptr = (npy_int32*) (pos->data + i * pos->strides[0]);
      jptr = (npy_int32*) (pos->data + i * pos->strides[0] + pos->strides[1]);

      *iptr = g_idxs[i] / w;
      *jptr = g_idxs[i] % w;
  }

  return_val = PyArray_Return(pos);
  return return_val;
}


// weights:        flattened h x w grid of costs
// h, w:           height and width of grid
// start, goal:    index of start/goal in flattened grid
// diag_ok:        if true, allows diagonal moves (8-conn.)
// paths (output): for each node, stores previous node in path
static PyObject *astar_path(PyObject *self, PyObject *args) {
  const PyArrayObject* weights_object;
  float dist_weight;
  int h;
  int w;
  int start;
  int goal;
  int diag_ok;

  if (!PyArg_ParseTuple(
        args, "Ofiiiii", // i = int, O = object
        &weights_object, &dist_weight,
        &h, &w,
        &start, &goal,
        &diag_ok))
    return NULL;

  float* weights = (float*) weights_object->data;
  int* paths = new int[h * w];
  int path_length = -1;

  Node start_node(start, start, weights[start], 0., 1);

  float* costs = new float[h * w];
  for (int i = 0; i < h * w; ++i)
    costs[i] = INF;
  costs[start] = weights[start];

  std::priority_queue<Node> nodes_to_visit;
  nodes_to_visit.push(start_node);

  int* nbrs = new int[8];

  // Normalize the distance metric
  float dist_norm = h + w;
  Node* closest = NULL;
  complete_path(weights, &start_node, w, h, goal);
  while (!nodes_to_visit.empty()) {
    // .top() doesn't actually remove the node
    Node cur = nodes_to_visit.top();
    nodes_to_visit.pop();


    // Keep track of the closest node to the goal
    if (closest == NULL || closest->h > cur.h) {
      closest = &cur;
    }

    if (cur.idx == goal) {
      path_length = cur.path_length;
      break;
    }

    int row = cur.idx / w;
    int col = cur.idx % w;
    // check bounds and find up to eight neighbors: top to bottom, left to right
    nbrs[0] = (diag_ok && row > 0 && col > 0)          ? cur.idx - w - 1   : -1;
    nbrs[1] = (row > 0)                                ? cur.idx - w       : -1;
    nbrs[2] = (diag_ok && row > 0 && col + 1 < w)      ? cur.idx - w + 1   : -1;
    nbrs[3] = (col > 0)                                ? cur.idx - 1       : -1;
    nbrs[4] = (col + 1 < w)                            ? cur.idx + 1       : -1;
    nbrs[5] = (diag_ok && row + 1 < h && col > 0)      ? cur.idx + w - 1   : -1;
    nbrs[6] = (row + 1 < h)                            ? cur.idx + w       : -1;
    nbrs[7] = (diag_ok && row + 1 < h && col + 1 < w ) ? cur.idx + w + 1   : -1;

    float heuristic_cost;
    for (int i = 0; i < 8; ++i) {
      if (nbrs[i] >= 0) {
        // The maximum value on the path
        float new_cost = std::max(cur.g, weights[nbrs[i]]);
        int max_g_idx = cur.max_g_idx;
        if (weights[nbrs[i]] > cur.g) {
          max_g_idx = nbrs[i];
        }
        if (new_cost < costs[nbrs[i]]) {
          // estimate the cost to the goal based on legal moves
          if (diag_ok) {
            heuristic_cost = dist_weight * linf_norm(nbrs[i] / w, nbrs[i] % w,
                                       goal    / w, goal    % w) / dist_norm;
          }
          else {
            heuristic_cost = dist_weight * l1_norm(nbrs[i] / w, nbrs[i] % w,
                                     goal    / w, goal    % w) / dist_norm;
          }

          // paths with lower expected cost are explored first
          nodes_to_visit.push(Node(nbrs[i], max_g_idx, new_cost, heuristic_cost, cur.path_length + 1));
          
          costs[nbrs[i]] = new_cost;
          paths[nbrs[i]] = cur.idx;
        }
      }
    }
  }
  
  PyObject *return_val;
  int idx = 0;
  if (path_length >= 0) {
    idx = goal;
  }
  else {
    idx = closest->idx;
  }

  npy_intp dims[2] = {path_length, 2};
  PyArrayObject* path = (PyArrayObject*) PyArray_SimpleNew(2, dims, NPY_INT32);
  npy_int32 *iptr, *jptr;
  for (npy_intp i = dims[0] - 1; i >= 0; --i) {
      iptr = (npy_int32*) (path->data + i * path->strides[0]);
      jptr = (npy_int32*) (path->data + i * path->strides[0] + path->strides[1]);

      *iptr = idx / w;
      *jptr = idx % w;

      idx = paths[idx];
  }

  return_val = PyArray_Return(path);

  delete[] costs;
  delete[] nbrs;
  delete[] paths;

  return return_val;
}

static PyMethodDef astar_methods[] = {
    {"astar_trials", (PyCFunction)astar_trials, METH_VARARGS, "astar_trials"},
    {"astar_path", (PyCFunction)astar_path, METH_VARARGS, "astar_path"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef astar_module = {
    PyModuleDef_HEAD_INIT,"astar", NULL, -1, astar_methods
};

PyMODINIT_FUNC PyInit_astar(void) {
  import_array();
  return PyModule_Create(&astar_module);
}

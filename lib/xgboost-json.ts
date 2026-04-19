/**
 * Pure TypeScript XGBoost JSON model inference engine.
 * Parses and evaluates XGBoost models saved in JSON format from Python.
 */

interface TreeNode {
  nodeid: number;
  depth?: number;
  split?: string;
  split_condition?: number;
  yes?: number;
  no?: number;
  missing?: number;
  leaf?: number;
  children?: TreeNode[];
}

interface Tree {
  id: number;
  nodes: TreeNode[];
}

interface XGBoostModelJSON {
  learner: {
    attributes?: Record<string, string>;
    feature_names?: string[];
    feature_types?: string[];
    gradient_booster: {
      model: {
        gbtree_model_param: {
          num_parallel_tree: string;
          num_trees: string;
        };
        trees: Tree[];
        tree_info?: number[];
      };
      name?: string;
    };
    learner_model_param?: {
      base_score?: string;
      num_class?: string;
      num_feature?: string;
    };
    objective?: {
      name: string;
      reg_loss_param?: Record<string, string>;
    };
  };
  version?: number[];
}

interface ParsedTree {
  nodes: Map<number, TreeNode>;
  rootId: number;
}

export class XGBoostJSONModel {
  private trees: ParsedTree[] = [];
  private baseScore: number = 0.5;
  private numClass: number = 1;
  private objective: string = "binary:logistic";
  private numFeatures: number = 0;

  constructor(modelJson: XGBoostModelJSON) {
    this.parseModel(modelJson);
  }

  private parseModel(json: XGBoostModelJSON): void {
    const learner = json.learner;
    const booster = learner.gradient_booster;
    const model = booster.model;

    // Parse learner params
    if (learner.learner_model_param) {
      this.baseScore = parseFloat(learner.learner_model_param.base_score || "0.5");
      this.numClass = parseInt(learner.learner_model_param.num_class || "0", 10);
      this.numFeatures = parseInt(learner.learner_model_param.num_feature || "0", 10);
    }

    // Parse objective
    if (learner.objective) {
      this.objective = learner.objective.name;
    }

    // Parse trees
    const numTrees = parseInt(model.gbtree_model_param.num_trees, 10);
    const treesData = model.trees;

    if (!treesData || treesData.length === 0) {
      throw new Error("No trees found in model");
    }

    for (let i = 0; i < numTrees; i++) {
      const treeData = treesData[i];
      const nodes = new Map<number, TreeNode>();

      // Build node map from the tree structure
      this.buildNodeMap(treeData, nodes);

      this.trees.push({
        nodes,
        rootId: 0,
      });
    }
  }

  private buildNodeMap(treeData: Tree, nodes: Map<number, TreeNode>): void {
    // XGBoost JSON format stores nodes in a flat array or nested structure
    // Handle the flat array format (most common in newer XGBoost versions)
    if (Array.isArray(treeData)) {
      // Flat array of nodes
      for (const node of treeData as unknown as TreeNode[]) {
        nodes.set(node.nodeid, node);
      }
    } else if (treeData.nodes) {
      // Nested format with nodes array
      for (const node of treeData.nodes) {
        nodes.set(node.nodeid, node);
        if (node.children) {
          for (const child of node.children) {
            this.addNodeRecursive(child, nodes);
          }
        }
      }
    }
  }

  private addNodeRecursive(node: TreeNode, nodes: Map<number, TreeNode>): void {
    nodes.set(node.nodeid, node);
    if (node.children) {
      for (const child of node.children) {
        this.addNodeRecursive(child, nodes);
      }
    }
  }

  private evaluateTree(tree: ParsedTree, features: number[]): number {
    let nodeId = tree.rootId;
    let iterations = 0;
    const maxIterations = 1000; // Safety limit

    while (iterations < maxIterations) {
      iterations++;
      const node = tree.nodes.get(nodeId);

      if (!node) {
        console.error(`Node ${nodeId} not found in tree`);
        return 0;
      }

      // Leaf node
      if (node.leaf !== undefined) {
        return node.leaf;
      }

      // Internal node - need to traverse
      if (node.split === undefined || node.split_condition === undefined) {
        // If no split info but has children, something is wrong
        console.error("Node missing split info:", node);
        return 0;
      }

      // Get feature index from split name (e.g., "f0" -> 0)
      const featureIdx = parseInt(node.split.replace("f", ""), 10);
      const featureValue = features[featureIdx] ?? 0;

      // Handle missing values
      if (featureValue === null || featureValue === undefined || Number.isNaN(featureValue)) {
        nodeId = node.missing ?? node.yes ?? 0;
      } else if (featureValue < node.split_condition) {
        nodeId = node.yes ?? 0;
      } else {
        nodeId = node.no ?? 0;
      }
    }

    console.error("Max iterations reached in tree evaluation");
    return 0;
  }

  /**
   * Predict raw scores for a single sample.
   * For binary classification, returns a single score.
   * For multi-class, returns an array of scores.
   */
  predict(features: number[]): number | number[] {
    if (this.numClass > 2) {
      // Multi-class classification
      const scores = new Array(this.numClass).fill(0);
      for (let i = 0; i < this.trees.length; i++) {
        const classIdx = i % this.numClass;
        scores[classIdx] += this.evaluateTree(this.trees[i], features);
      }
      return scores;
    } else {
      // Binary classification or regression
      let score = this.baseScore;
      for (const tree of this.trees) {
        score += this.evaluateTree(tree, features);
      }
      return score;
    }
  }

  /**
   * Predict probability for binary classification.
   */
  predictProba(features: number[]): number {
    const score = this.predict(features);
    if (typeof score === "number") {
      // Apply sigmoid for binary classification
      return 1 / (1 + Math.exp(-score));
    }
    throw new Error("predictProba is for binary classification only");
  }

  /**
   * Predict probabilities for multi-class classification.
   */
  predictProbaMulti(features: number[]): number[] {
    const scores = this.predict(features);
    if (Array.isArray(scores)) {
      // Apply softmax
      const maxScore = Math.max(...scores);
      const expScores = scores.map((s) => Math.exp(s - maxScore));
      const sum = expScores.reduce((a, b) => a + b, 0);
      return expScores.map((s) => s / sum);
    }
    throw new Error("predictProbaMulti is for multi-class classification only");
  }

  getNumTrees(): number {
    return this.trees.length;
  }

  getNumClass(): number {
    return this.numClass;
  }

  getObjective(): string {
    return this.objective;
  }
}

/**
 * Alternative parser for the newer XGBoost JSON format that uses a different structure.
 * This handles models where nodes are stored in a flattened array format.
 */
interface FlatNode {
  categories?: number[];
  categories_nodes?: number[];
  categories_segments?: number[];
  categories_sizes?: number[];
  default_left?: boolean[];
  hess?: number[];
  left_children?: number[];
  loss_changes?: number[];
  parents?: number[];
  right_children?: number[];
  split_conditions?: number[];
  split_indices?: number[];
  split_type?: number[];
  sum_hess?: number[];
  base_weights?: number[];
}

interface FlatTreeModel {
  learner: {
    attributes?: Record<string, string>;
    feature_names?: string[];
    feature_types?: string[];
    gradient_booster: {
      model: {
        gbtree_model_param: {
          num_parallel_tree: string;
          num_trees: string;
        };
        iteration_indptr?: number[];
        trees?: FlatNode[];
        tree_info?: number[];
      };
      name?: string;
    };
    learner_model_param?: {
      base_score?: string;
      num_class?: string;
      num_feature?: string;
    };
    objective?: {
      name: string;
      reg_loss_param?: Record<string, string>;
    };
  };
  version?: number[];
}

interface ParsedFlatTree {
  leftChildren: number[];
  rightChildren: number[];
  splitIndices: number[];
  splitConditions: number[];
  baseWeights: number[];
  defaultLeft: boolean[];
}

export class XGBoostFlatModel {
  private trees: ParsedFlatTree[] = [];
  private baseScore: number = 0.5;
  private numClass: number = 1;
  private objective: string = "binary:logistic";

  constructor(modelJson: FlatTreeModel) {
    this.parseModel(modelJson);
  }

  private parseModel(json: FlatTreeModel): void {
    const learner = json.learner;
    const booster = learner.gradient_booster;
    const model = booster.model;

    // Parse learner params
    if (learner.learner_model_param) {
      this.baseScore = parseFloat(learner.learner_model_param.base_score || "0.5");
      this.numClass = parseInt(learner.learner_model_param.num_class || "0", 10);
    }

    // Parse objective
    if (learner.objective) {
      this.objective = learner.objective.name;
    }

    // Parse trees from flat format
    const treesData = model.trees;
    if (!treesData || treesData.length === 0) {
      throw new Error("No trees found in model");
    }

    for (const treeData of treesData) {
      this.trees.push({
        leftChildren: treeData.left_children || [],
        rightChildren: treeData.right_children || [],
        splitIndices: treeData.split_indices || [],
        splitConditions: treeData.split_conditions || [],
        baseWeights: treeData.base_weights || [],
        defaultLeft: treeData.default_left || [],
      });
    }
  }

  private evaluateTree(tree: ParsedFlatTree, features: number[]): number {
    let nodeIdx = 0;
    let iterations = 0;
    const maxIterations = 1000;

    while (iterations < maxIterations) {
      iterations++;

      const leftChild = tree.leftChildren[nodeIdx];

      // Leaf node (left_child == -1 indicates leaf)
      if (leftChild === -1) {
        return tree.baseWeights[nodeIdx];
      }

      const featureIdx = tree.splitIndices[nodeIdx];
      const threshold = tree.splitConditions[nodeIdx];
      const featureValue = features[featureIdx] ?? 0;

      // Handle missing values
      if (featureValue === null || featureValue === undefined || Number.isNaN(featureValue)) {
        nodeIdx = tree.defaultLeft[nodeIdx] ? leftChild : tree.rightChildren[nodeIdx];
      } else if (featureValue < threshold) {
        nodeIdx = leftChild;
      } else {
        nodeIdx = tree.rightChildren[nodeIdx];
      }
    }

    console.error("Max iterations reached");
    return 0;
  }

  predict(features: number[]): number | number[] {
    if (this.numClass > 2) {
      const scores = new Array(this.numClass).fill(0);
      for (let i = 0; i < this.trees.length; i++) {
        const classIdx = i % this.numClass;
        scores[classIdx] += this.evaluateTree(this.trees[i], features);
      }
      return scores;
    } else {
      let score = this.baseScore;
      for (const tree of this.trees) {
        score += this.evaluateTree(tree, features);
      }
      return score;
    }
  }

  predictProba(features: number[]): number {
    const score = this.predict(features);
    if (typeof score === "number") {
      return 1 / (1 + Math.exp(-score));
    }
    throw new Error("predictProba is for binary classification only");
  }

  predictProbaMulti(features: number[]): number[] {
    const scores = this.predict(features);
    if (Array.isArray(scores)) {
      const maxScore = Math.max(...scores);
      const expScores = scores.map((s) => Math.exp(s - maxScore));
      const sum = expScores.reduce((a, b) => a + b, 0);
      return expScores.map((s) => s / sum);
    }
    throw new Error("predictProbaMulti is for multi-class classification only");
  }

  getNumClass(): number {
    return this.numClass;
  }
}

/**
 * Load an XGBoost model from JSON, automatically detecting the format.
 */
export function loadXGBoostModel(
  modelJson: XGBoostModelJSON | FlatTreeModel
): XGBoostJSONModel | XGBoostFlatModel {
  const trees = modelJson.learner?.gradient_booster?.model?.trees;

  if (!trees || trees.length === 0) {
    throw new Error("No trees found in model JSON");
  }

  // Check the format by examining the first tree
  const firstTree = trees[0];

  // Flat format has left_children array
  if ("left_children" in firstTree) {
    return new XGBoostFlatModel(modelJson as FlatTreeModel);
  }

  // Standard format has nodes array or is a tree structure
  return new XGBoostJSONModel(modelJson as XGBoostModelJSON);
}

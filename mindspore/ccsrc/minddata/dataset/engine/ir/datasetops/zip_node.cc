/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "minddata/dataset/engine/ir/datasetops/zip_node.h"

#include <memory>
#include <string>
#include <vector>

#include "minddata/dataset/engine/datasetops/zip_op.h"
#include "minddata/dataset/engine/opt/pass.h"
#include "minddata/dataset/util/status.h"
namespace mindspore {
namespace dataset {

ZipNode::ZipNode(const std::vector<std::shared_ptr<DatasetNode>> &datasets) {
  for (auto const &child : datasets) AddChild(child);
}

std::shared_ptr<DatasetNode> ZipNode::Copy() {
  std::vector<std::shared_ptr<DatasetNode>> empty_vector;
  empty_vector.clear();
  auto node = std::make_shared<ZipNode>(empty_vector);
  return node;
}

void ZipNode::Print(std::ostream &out) const { out << Name(); }

Status ZipNode::ValidateParams() {
  if (children_.size() < 2) {
    std::string err_msg = "ZipNode: input datasets are not specified.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  if (find(children_.begin(), children_.end(), nullptr) != children_.end()) {
    std::string err_msg = "ZipNode: input datasets should not be null.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::vector<std::shared_ptr<DatasetOp>> ZipNode::Build() {
  // A vector containing shared pointer to the Dataset Ops that this object will create
  std::vector<std::shared_ptr<DatasetOp>> node_ops;

  node_ops.push_back(std::make_shared<ZipOp>(rows_per_buffer_, connector_que_size_));
  return node_ops;
}

// Visitor accepting method for NodePass
Status ZipNode::Accept(NodePass *p, bool *modified) {
  // Downcast shared pointer then call visitor
  return p->Visit(shared_from_base<ZipNode>(), modified);
}

// Visitor accepting method for NodePass
Status ZipNode::AcceptAfter(NodePass *p, bool *modified) {
  // Downcast shared pointer then call visitor
  return p->VisitAfter(shared_from_base<ZipNode>(), modified);
}

}  // namespace dataset
}  // namespace mindspore
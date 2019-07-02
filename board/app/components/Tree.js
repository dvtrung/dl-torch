// @flow
import React, { Component } from 'react';
import { Link } from 'react-router-dom';
import routes from '../constants/routes';
import { Statistic, Row, Col, Icon, Button } from 'antd';
import { Tree } from 'antd';
import { onModelSelected } from '../actions/home';
import { connect } from 'react-redux';
import * as HomeActions from '../actions/home';
import { bindActionCreators } from 'redux';
const { TreeNode, DirectoryTree } = Tree;

type Props = {
  dirs: [any]
};

class ExplorerTree extends Component<Props> {
  props: Props;

  onSelect = ([key]: string, event) => {
    const root = key.split(',')[0];
    const modelName = key.split(',')[1];
    this.props.onModelSelected(root, modelName);
  };

  onExpand = () => {
    console.log('Trigger Expand');
  };

  onAddFolder = () => {
    console.log('Add Folder');
  }

  render() {
    return (
      <div style={{height:"100%"}}>
        <Button onClick={this.props.loadDirsAsync}>Refresh</Button>
        <Button onClick={this.onAddFolder}>Add Folder</Button>
        { this.props.dirs ? (
          <DirectoryTree theme="dark" multiple defaultExpandAll onSelect={this.onSelect} onExpand={this.onExpand}>
            {Object.entries(this.props.dirs).map(([path, dir]) =>
            <TreeNode title={dir.name} key={path}>
              {Object.entries(dir.models).map(([modelName, model]) =>
                <TreeNode title={modelName} key={[path, modelName]} isLeaf />)}
            </TreeNode>)}
          </DirectoryTree>) : (<div>No directory.</div>)
        }
      </div>
    );
  }
}

const mapDispatchToProps = (dispatch) => {
  return bindActionCreators({
    ...HomeActions
  }, dispatch)
}

export default connect(
  null,
  mapDispatchToProps
)(ExplorerTree);

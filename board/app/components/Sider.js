// @flow
import React, { Component } from 'react';
import { Icon, Layout, Menu } from 'antd';
import {connect} from "react-redux";
import type {Model} from "../reducers/types";
import {bindActionCreators} from "redux";
import * as HomeActions from "../actions/home";
import * as SettingActions from "../actions/settings";
import {SELECT_MODEL} from "../actions/home";
import {onModelSelected} from "../actions/home";
import {onMachineSelected} from "../actions/home";

const { SubMenu } = Menu;
type Props = {
  models: [Model],
  selectModel: (string) => void
};

class Sider extends Component<Props> {
  render() {
    return (
      <Layout.Sider style={{background: '#fff', height: "100%", overflow: "auto"}}>
        <Menu
          theme="dark"
          defaultSelectedKeys={['1']}
          defaultOpenKeys={['models', 'machines']}
          mode="inline"
          style={{height: "100%"}}>
          <SubMenu
            key="models"
            title={
              <span>
                <Icon type="user"/>
                <span>Models</span>
              </span>
            }
          >
            {Object.entries(this.props.models).map(([key, model]) =>
              <Menu.Item key={`model-${key}`} onClick={() => this.props.selectModel(key)}>{key}</Menu.Item>)}
          </SubMenu>
          <SubMenu
            key="machines"
            title={
              <span>
                <Icon type="team"/>
                <span>Machines</span>
              </span>
            }
          >
            {Object.entries(this.props.machines).map(([key, machine]) =>
              <Menu.Item key={`machine-${key}`} onClick={() => this.props.selectMachine(key)}>{key}</Menu.Item>
            )}
          </SubMenu>
          <Menu.Item key="9">
            <Icon type="file"/>
            <span>File</span>
          </Menu.Item>
        </Menu>
      </Layout.Sider>
    );
  }
}

function mapStateToProps(state) {
  return {
    models: state.settings.models,
    machines: state.settings.machines
  };
}

function mapDispatchToProps(dispatch) {
  return {
    selectModel: bindActionCreators(onModelSelected, dispatch),
    selectMachine: bindActionCreators(onMachineSelected, dispatch)
  }
}

export default connect(
  mapStateToProps,
  mapDispatchToProps
)(Sider);

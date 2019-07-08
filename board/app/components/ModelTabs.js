// @flow
import React, { Component } from 'react';
import {Button, Col, Form, Icon, Input, Layout, Menu, Row, Tabs} from 'antd';
import {connect} from "react-redux";
import type {Machine, Model} from "../reducers/types";
import {bindActionCreators} from "redux";
import * as HomeActions from "../actions/home";
import * as SettingActions from "../actions/settings";
import {SELECT_MODEL} from "../actions/home";
import Stats from "./Stats";
import MachineSelect from "./MachineSelect";
import Terminal from "./Terminal";
import {getLogPath, sshConnect} from "../utils/ssh";
import {ipcRenderer} from "electron";
import {syncEpochStats} from "../actions/home";
import TrainingTab from "./tabs/TrainingTab";

const { TabPane }= Tabs;
const { TextArea } = Input;

type Props = {
  models: [Model],
  selectModel: (Model) => void,
  syncEpochStats: (Model, Machine) => void
};

const formItemLayout = {
  labelCol: {
    xs: { span: 24 },
    sm: { span: 4 },
  },
  wrapperCol: {
    xs: { span: 24 },
    sm: { span: 20 },
  },
};

class ModelDetailsForm extends Component {
  handleSubmit = () => {

  }

  render() {
    const { model, machine } = this.props;
    const { getFieldDecorator } = this.props.form;
    return (
      <Form {...formItemLayout} onSubmit={this.handleSubmit}>
        <Form.Item label="Path">
          <Input value={model.path} />
        </Form.Item>
        <Form.Item label="Machine">
          <MachineSelect value={model.machine} onChange={() => {}} />
        </Form.Item>
        <Form.Item label="Remote Directory">
          <Input value={model.remoteDir} />
        </Form.Item>
      </Form>
    )
  }
}

const ModelDetails = Form.create({ name: 'register' })(ModelDetailsForm);

class ModelTabs extends Component<Props> {
  render() {
    const { model, machine } = this.props;
    return (
      <Tabs defaultActiveKey="details" onChange={this.onTabChange}>
        <TabPane tab="Details" key="details">
          <ModelDetails model={model} machine={machine} />
        </TabPane>
        <TabPane tab="Training" key="training">
          <TrainingTab />
        </TabPane>
        <TabPane tab="Configuration" key="config">
          <Row style={{height:"100vh"}}>
            <Col span={8}>
              <TextArea value={this.props.selectedModel != null ? this.props.selectedModel.config : ""}></TextArea>
            </Col>
          </Row>
        </TabPane>
      </Tabs>
    );
  }
}

function mapStateToProps(state) {
  const model = state.home.models[state.home.selectedModelKey];
  return {
    model,
    machine: state.settings.machines[model.machine]
  };
}

function mapDispatchToProps(dispatch) {
  return {
    selectModel: (model) => {
      return dispatch({
        type: SELECT_MODEL,
        model: model
      })
    },
    syncEpochStats
  }
}

export default connect(
  mapStateToProps,
  mapDispatchToProps
)(ModelTabs);

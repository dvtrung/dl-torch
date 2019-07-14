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
import EvaluationTab from "./tabs/EvaluationTab";
import ModelsTab from "./tabs/ModelsTab";

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

class MachineDetailsForm extends Component {
  handleSubmit = () => {

  }

  render() {
    const { machine } = this.props;
    const { getFieldDecorator } = this.props.form;
    return (
      <Form {...formItemLayout} onSubmit={this.handleSubmit}>
        <Form.Item label="Name">
          <Input value={machine.key} />
        </Form.Item>
        <Form.Item label="Host">
          <Input value={machine.host} onChange={() => {}} />
        </Form.Item>
        <Form.Item label="Root">
          <Input value={machine.root} />
        </Form.Item>
        <Form.Item label="Username">
          <Input value={machine.username} />
        </Form.Item>
        <Form.Item label="Password">
          <Input value={machine.password} />
        </Form.Item>
        <Form.Item label="Private Key">
          <TextArea value={machine.privateKey} rows={5} />
        </Form.Item>
      </Form>
    )
  }
}

const MachineDetails = Form.create({ name: 'register' })(MachineDetailsForm);

class MachineTabs extends Component<Props> {
  render() {
    const { model, machine } = this.props;
    return (
      <Tabs defaultActiveKey="details" onChange={this.onTabChange}>
        <TabPane tab={<span><Icon type="info" /> Details</span>} key="details">
          <MachineDetails machine={machine} />
        </TabPane>
        <TabPane tab={<span><Icon type="line-chart" /> Models</span>} key="models">
          <ModelsTab />
        </TabPane>
      </Tabs>
    );
  }
}

function mapStateToProps(state) {
  const machine = state.home.machines[state.home.selectedMachineKey];
  return {
    machine
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
)(MachineTabs);

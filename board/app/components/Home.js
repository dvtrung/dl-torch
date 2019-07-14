// @flow
import React, { Component } from 'react';
import { Layout, Button, Input, Row, Col, Tabs } from 'antd';
import Stats from './Stats'
import Sider from './Sider'
import ModelTabs from "./ModelTabs";
import MachineTabs from "./MachineTabs";

const { Content } = Layout;
const { TabPane } = Tabs;

type Props = {
  loadModelsAsync: (path: string) => void,
  loadSettings: () => void,
  currentDir: string,
  selectedModel: any,
  machines: [any],
  dirs: [string],
  settings: any
};

export default class Home extends Component<Props> {
  state = {
    selectedMachineKey: "aws",
    selectedMachine: this.props.machines["aws"]
  };

  constructor(props) {
    super(props)
  }

  componentDidMount() {
    this.props.loadSettings();
    this.props.loadModelsAsync("C:\\repos\\dvtrung\\dlex\\implementations\\speech_recognition")
    this.props.loadModelsAsync("C:\\repos\\dvtrung\\dlex\\implementations\\cnn")
  }

  onTabChange() {

  };

  onMachineChange = (key) => {
    this.setState({
      selectedMachineKey: key,
      selectedMachine: this.props.machines[key]
    })
  };

  render() {
    return (
      <Layout style={{height:"100vh"}}>
        <Sider />
        <Layout>
          <Content style={{padding: 24}}>
            {this.props.selectedModelKey && <ModelTabs />}
            {this.props.selectedMachineKey && <MachineTabs />}
          </Content>
        </Layout>
      </Layout>
    );
  }
}


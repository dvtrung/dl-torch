// @flow
import React, { Component } from 'react';
import { Layout, Button, Input, Row, Col, Tabs } from 'antd';
import Stats from './Stats'
import Tree from './Tree'
import MachineSelect from './MachineSelect'
import Terminal from './Terminal'

const { Sider } = Layout;
const { TextArea } = Input;
const { TabPane } = Tabs;

type Props = {
  loadModelsAsync: (path: string) => void,
  loadSettings: () => void,
  currentDir: string,
  selectedModel: any,
  dirs: [string],
  settings: any
};

export default class Home extends Component<Props> {
  state = {
    selectedMachine: "peterchin3"
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

  }

  onMachineChange(key) {
    this.setState({
      selectedMachine: key
    })
  }

  render() {
    return (
      <Layout style={{height:"100vh"}}>
        <Sider style={{ background: '#fff', height:"100%", overflow: "auto" }}>
          <Tree dirs={this.props.dirs}/>
        </Sider>
        <Layout>
          <Tabs defaultActiveKey="1" onChange={this.onTabChange}>
            <TabPane tab="Overview" key="1">
              <Stats epoch={14} totalEpoch={100} trainError={5.83} testError={4.27} />
              <MachineSelect value={this.state.selectedMachine} handleChange={this.onMachineChange}/>
              <Terminal machine={this.props.settings.machines[this.state.selectedMachine]}/>
            </TabPane>
            <TabPane tab="Config" key="2">
            <Row style={{height:"100vh"}}>
            <Col span={8}>
              <TextArea value={this.props.selectedModel != null ? this.props.selectedModel.config : ""}></TextArea>
            </Col>
          </Row>
            </TabPane>
          </Tabs>
        </Layout>
      </Layout>
    );
  }
}


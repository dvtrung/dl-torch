// @flow
import React, { Component } from 'react';
import { Layout, Button, Input, Row, Col, Tabs, Icon } from 'antd';
import type {Machine} from "../reducers/types";
import {bindActionCreators} from "redux";
import {connect} from "react-redux";
import {sshConnect} from "../utils/ssh";
import {ipcRenderer} from "electron";
const { Sider, Content } = Layout;
const { TextArea } = Input;
const { TabPane } = Tabs;

type Props = {
  machine: Machine
};

class Terminal extends Component<Props> {
  constructor(props) {
    super(props)
    this.state = {
      client: null,
      isConnecting: false,
      output: []
    }
  }

  componentDidMount() {

  }

  onConnect = () => {
    this.setState({ isConnecting: true });
    sshConnect(this.props.machine, (client) => {
      this.setState({ client, isConnecting: false });
      const model = { name: "iwslt15_en_vi" };
      // client.exec(`tail -f ${this.props.machine.tmpPath}/nohups/iwslt15_en_vi.out`, (err, stream) => {
      const logDir = `${this.props.machine.root}/nmt/logs/${model.name}`
      client.exec(`tail -f ${logDir}/\$(ls ${logDir} -S | tail -n 1)/results.json`, (err, stream) => {
        if (err) throw err;
        stream.on('close', (code, signal) => {
          console.log('Stream :: close :: code: ' + code + ', signal: ' + signal);
          client.end();
        }).on('data', (data) => {
          this.setState({
            output: data.toString()
          })
        }).stderr.on('data', (data) => {
          console.log('STDERR: ' + data);
        });
      });
    }, (err) => {
      ipcRenderer.send("error-box", {
        message: err.toString()
      });
    });
  };

  onDisconnect = () => {
    this.state.client.end();
    this.setState({ client: null })
  };

  render() {
    const isConnected = this.state.client != null
    const btnConnect = isConnected ?
      <Button type="danger" onClick={this.onDisconnect}><Icon type="disconnect" /> Disconnect</Button> :
      (this.state.isConnecting ?
        <Button type="primary" disabled={true}><Icon type="sync" spin /> Connecting...</Button> :
        <Button type="primary" onClick={this.onConnect}><Icon type="up-circle" /> Connect</Button>)

    return <div>
      {btnConnect}
      <div>{this.state.output}</div>
      <span>
        {isConnected ? "Connected" : (this.state.isConnecting ? "Connecting..." : "Not connected")}</span>
    </div>
  }
}

function mapStateToProps(state) {
  return {

  };
}

const mapDispatchToProps = dispatch => bindActionCreators({}, dispatch);

export default connect(
  mapStateToProps,
  mapDispatchToProps
)(Terminal);


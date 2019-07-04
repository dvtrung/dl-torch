// @flow
import React, { Component } from 'react';
import { Link } from 'react-router-dom';
import routes from '../constants/routes';
import styles from './Home.css';
import { Statistic, Row, Col, Icon } from 'antd';
type Props = {
  stats: any,
};

export default class Stats extends Component<Props> {
  constructor() {
    super();
    this.state = {
      currentMetric: 0
    }
  }

  onSelect = (keys, event) => {
    console.log('Trigger Select', keys, event);
  };

  onExpand = () => {
    console.log('Trigger Expand');
  };

  render() {
    const { stats } = this.props;
    const metric = stats.metrics[this.state.currentMetric]
    return (
      <Row gutter={16}>
        <Col span={8}>
          <Statistic title="Epoch" value={stats.epoch} suffix={"/ " + stats.totalEpoch} />
        </Col>
        <Col span={8}>
          <Statistic title={`Train ${metric}`} value={(stats.bestResult[this.state.currentMetric] * 100).toFixed(4).toString()} />
        </Col>
        <Col span={8}>
          <Statistic title={`Test ${metric}`} value={(stats.bestResult[this.state.currentMetric] * 100).toFixed(4).toString()} />
        </Col>
      </Row>
    );
  }
}


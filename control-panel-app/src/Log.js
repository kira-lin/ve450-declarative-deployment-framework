import React, { Component } from 'react';
import './App.css';

import ReactTable from "react-table";
import "react-table/react-table.css";
import Tree from "react-tree-graph";
import "react-tree-graph/dist/style.css";

class Log extends Component {

  componentWillMount() {
    this.setState({
      log_list: [
        { "jobId": "1005", "status": "pending", "time": "2018-10-06 05:05:05" },
        { "jobId": "1004", "status": "running", "time": "2018-10-05 04:04:04" },
        { "jobId": "1003", "status": "done", "time": "2018-10-04 03:03:03" },
        { "jobId": "1002", "status": "done", "time": "2018-10-03 02:02:02" },
        { "jobId": "1001", "status": "error", "time": "2018-10-02 01:01:01" },
        { "jobId": "1000", "status": "done", "time": "2018-10-01 00:00:00" }
      ]
    })
  }

  constructor(props) {
    // Initialize the states
    super(props);
    this.state = {
      "log_list": [],
      "tree_node": {}
    };
  }

  render() {

    return (
      <div>
        <ReactTable
          data={this.state["log_list"]}
          columns={[
            {
              Header: "Job ID",
              accessor: "jobId"
            },
            {
              Header: "Status",
              accessor: "status",
              Cell: row => (
                <span>
                  <span style={{
                    color: row.value === 'error' ? '#ff2e00'
                         : row.value === 'pending' ? '#ffbf00'
                         : row.value === 'running' ? '#ffbf00'
                          : '#57d500',
                    transition: 'all .3s ease'
                  }}>
                    &#x25cf;
                  </span> {
                    row.value
                  }
                </span>
              )
            },
            {
              Header: "Time",
              accessor: "time"
            }
          ]}
          getTrProps={(state, rowInfo, column, instance) => ({
            onClick: e => {
              this.setState({
                "tree_node": {
                  name: 'run_this_last',
                  children: [
                    {
                      name: 'run_this',
                      children: [
                        {
                          name: 'run_1',
                          children: [
                            {name: 'first_1'}
                          ]},
                        {
                          name: 'run_2'
                        }
                      ]
                    },
                    {name: 'also_run_this'}]
                }
              });
              return true;
            }
          })}
          defaultSorted={[
            {
              id: "jobId",
              desc: true
            }
          ]}
          defaultPageSize={5}
          className="-striped -highlight"
        />
        <Tree
          data={this.state["tree_node"]}
          height={400}
          width={400}>
          <text
            dy="100"
            dx="5"
            stroke="#000000"
            fill="#000000">
            Tree View
          </text>
        </Tree>
      </div>
    );

  }

}


export default Log;
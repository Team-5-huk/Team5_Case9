import React from 'react';
import { Box } from '@mui/material';
import { DataGrid } from '@mui/x-data-grid';

const ListCompleted = ({columns, data, handleTableItem}) => {
	return (
			<Box
				m='20px 0'
				height='50vh'
				borderRadius={5}
				sx={{
					'& .MuiDataGrid-root': {
						border: '1px solid #99cbff',
						borderRadius: 5,
					},
					'& .MuiDataGrid-cell': {
						borderBottom: 'none',
					},
					'& .name-column--cell': {
						// color: '#9e9e9e',
					},
					'& .MuiDataGrid-columnHeaders': {
						backgroundColor: '#99cbff',
						borderBottom: 'none',
					},
					'& .MuiDataGrid-virtualScroller': {
						// backgroundColor: '#b9d4f0',
					},
					'& .MuiDataGrid-footerContainer': {
						borderTop: 'none',
						// backgroundColor: '#b9d4f0',
					},
					'& .MuiCheckbox-root': {
						color: '#b9d4f0',
					},
				}}
			>
				<DataGrid 
					columns={columns} 
					rows={data} 
					onRowClick={handleTableItem}
				/>
			</Box>
	);
};

export default ListCompleted;

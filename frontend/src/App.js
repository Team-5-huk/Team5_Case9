import styles from './App.module.scss';
import { observer } from 'mobx-react-lite';
import store from './store/store';
import NoAccessRole from './routes/NoAccessRole';
import EmployeeRoutes from './routes/EmployeeRoutes';
import AdminRoutes from './routes/AdminRoutes';
import ClientRoutes from './routes/ClientRoutes';

const App = observer(() => {
	const { authUser } = store;

	const routes = {
		logout: <NoAccessRole />,
		employee: <EmployeeRoutes />,
		admin: <AdminRoutes />,
		user: <ClientRoutes />,
	};

	return <div className={styles.app}>{routes[authUser]}</div>;
});

export default App;
